/***************************************************************************
*                 Lempel, Ziv, Storer, and Szymanski Encoding
*
*   File    : lzss.c
*   Purpose : Use lzss coding (Storer and Szymanski's modified LZ77) to
*             compress lzss data files.
*   Author  : Michael Dipperstein
*   Date    : November 28, 2014
*
****************************************************************************
*
* LZss: An ANSI C LZSS Encoding/Decoding Routines
* Copyright (C) 2003 - 2007, 2014 by
* Michael Dipperstein (mdipper@alumni.engr.ucsb.edu)
*
* This file is part of the lzss library.
*
* The lzss library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of the
* License, or (at your option) any later version.
*
* The lzss library is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
* General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
***************************************************************************/

/***************************************************************************
*                             INCLUDED FILES
***************************************************************************/
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "lzlocal.h"
#include "lzss.h"
#include "statistics.h"
#include "bitfile.h"
#include <iostream>

/***************************************************************************
*                            TYPE DEFINITIONS
***************************************************************************/

/***************************************************************************
*                                CONSTANTS
***************************************************************************/

/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/
/* cyclic buffer sliding window of already read characters */
unsigned char slidingWindow[WINDOW_SIZE];
unsigned char uncodedLookahead[MAX_CODED];

/***************************************************************************
*                               PROTOTYPES
***************************************************************************/

/***************************************************************************
*                                FUNCTIONS
***************************************************************************/

/****************************************************************************
*   Function   : EncodeLZSS
*   Description: This function will read an input file and write an output
*                file encoded according to the traditional LZSS algorithm.
*                This algorithm encodes strings as 16 bits (a 12 bit offset
*                + a 4 bit length).
*   Parameters : fpIn - pointer to the open binary file to encode
*                fpOut - pointer to the open binary file to write encoded
*                       output
*   Effects    : fpIn is encoded and written to fpOut.  Neither file is
*                closed after exit.
*   Returned   : 0 for success, -1 for failure.  errno will be set in the
*                event of a failure.
****************************************************************************/

int EncodeLZSS(FileStream *fpIn, FileStream *fpOut, AppStatistics *metrics)
{
	bit_file_t *bfpOut;
	encoded_string_t matchData;
	int c;
	unsigned int i;
	unsigned int len; /* length of string */

	/* head of sliding window and lookahead */
	unsigned int windowHead, uncodedHead;

	/* validate arguments */
	if ((NULL == fpIn) || (NULL == fpOut))
	{
		errno = ENOENT;
		return -1;
	}

	/* convert output file to bitfile */
	// bfpOut = MakeBitFile(fpOut, BF_WRITE);

	// if (NULL == bfpOut)
	// {
	// 	perror("Making Output File a BitFile");
	// 	return -1;
	// }

	windowHead = 0;
	uncodedHead = 0;

	/************************************************************************
	* Fill the sliding window buffer with some known vales.  DecodeLZSS must
	* use the same values.  If common characters are used, there's an
	* increased chance of matching to the earlier strings.
	************************************************************************/
	memset(slidingWindow, ' ', WINDOW_SIZE * sizeof(unsigned char));

	/************************************************************************
	* Copy MAX_CODED bytes from the input file into the uncoded lookahead
	* buffer.
	************************************************************************/
	for (len = 0; len < MAX_CODED && (c = fpIn->GetChar()) != EOF; len++)
	{
		uncodedLookahead[len] = c;
	}

	if (0 == len)
	{
		return 0; /* inFile was empty */
	}

	/* Look for matching string in sliding window */
	i = InitializeSearchStructures();

	if (0 != i)
	{
		return i; /* InitializeSearchStructures returned an error */
	}
	metrics->StartFindMatch();
	matchData = FindMatch(windowHead, uncodedHead);
	metrics->StopFindMatch();
#ifdef PRINTOFFSETS
	printf("%i %i\n", matchData.offset, matchData.length);
#endif
	/* now encoded the rest of the file until an EOF is read */
	while (len > 0)
	{
		// printf("THISMETTERS %i %i\n",matchData.offset,matchData.offset);
		if (matchData.length > len)
		{
			/* garbage beyond last data happened to extend match length */
			matchData.length = len;
		}

		if (matchData.length <= MAX_UNCODED)
		{
			/* not long enough match.  write uncoded flag and character */
			fpOut->PutBit(UNCODED);
			fpOut->PutChar(uncodedLookahead[uncodedHead]);

			matchData.length = 1; /* set to 1 for 1 byte uncoded */
		}
		else
		{
			unsigned int adjustedLen;

			/* adjust the length of the match so minimun encoded len is 0*/
			adjustedLen = matchData.length - (MAX_UNCODED + 1);

			/* match length > MAX_UNCODED.  Encode as offset and length. */
			fpOut->PutBit(ENCODED);
			fpOut->PutBitsNum(&matchData.offset, OFFSET_BITS,
							  sizeof(unsigned int));
			fpOut->PutBitsNum(&adjustedLen, LENGTH_BITS,
							  sizeof(unsigned int));
		}

		/********************************************************************
		* Replace the matchData.length worth of bytes we've matched in the
		* sliding window with new bytes from the input file.
		********************************************************************/
		i = 0;
		while ((i < matchData.length) && ((c = fpIn->GetChar()) != EOF))
		{
			/* add old byte into sliding window and new into lookahead */
			ReplaceChar(windowHead, uncodedLookahead[uncodedHead]);
			uncodedLookahead[uncodedHead] = c;
			windowHead = Wrap((windowHead + 1), WINDOW_SIZE);
			uncodedHead = Wrap((uncodedHead + 1), MAX_CODED);
			i++;
		}

		/* handle case where we hit EOF before filling lookahead */
		while (i < matchData.length)
		{
			ReplaceChar(windowHead, uncodedLookahead[uncodedHead]);
			/* nothing to add to lookahead here */
			windowHead = Wrap((windowHead + 1), WINDOW_SIZE);
			uncodedHead = Wrap((uncodedHead + 1), MAX_CODED);
			len--;
			i++;
		}

		/* find match for the remaining characters */
		metrics->StartFindMatch();

		matchData = FindMatch(windowHead, uncodedHead);
		metrics->StopFindMatch();

#ifdef PRINTOFFSETS
		printf("%i %i\n", matchData.offset, matchData.length);
#endif
	}

	/* we've encoded everything, free bitfile structure */
	// BitFileToFILE(bfpOut);
	fpOut->Flush();

	return 0;
}

/****************************************************************************
*   Function   : DecodeLZSSByFile
*   Description: This function will read an LZSS encoded input file and
*                write an output file.  This algorithm encodes strings as 16
*                bits (a 12 bit offset + a 4 bit length).
*   Parameters : fpIn - pointer to the open binary file to decode
*                fpOut - pointer to the open binary file to write decoded
*                       output
*   Effects    : fpIn is decoded and written to fpOut.  Neither file is
*                closed after exit.
*   Returned   : 0 for success, -1 for failure.  errno will be set in the
*                event of a failure.
****************************************************************************/
int DecodeLZSS(FileStream *fpIn, FILE *fpOut)
{
	// bit_file_t *bfpIn;
	int c;
	unsigned int i, nextChar;
	encoded_string_t code; /* offset/length code for string */

	/* use stdin if no input file */
	if ((NULL == fpIn) || (NULL == fpOut))
	{
		errno = ENOENT;
		return -1;
	}

	/* convert input file to bitfile */
	// bfpIn = MakeBitFile(fpIn, BF_READ);

	// if (NULL == bfpIn)
	// {
	// 	perror("Making Input File a BitFile");
	// 	return -1;
	// }

	/************************************************************************
	* Fill the sliding window buffer with some known vales.  EncodeLZSS must
	* use the same values.  If common characters are used, there's an
	* increased chance of matching to the earlier strings.
	************************************************************************/
	memset(slidingWindow, ' ', WINDOW_SIZE * sizeof(unsigned char));

	nextChar = 0;

	while (1)
	{
		// if ((c = BitFileGetBit(bfpIn)) == EOF)
		if ((c = fpIn->GetBit()) == EOF)
		{
			/* we hit the EOF */
			break;
		}

		if (c == UNCODED)
		{
			/* uncoded character */
			if ((c = fpIn->BitGetChar()) == EOF)
			// if ((c = BitFileGetChar(bfpIn)) == EOF)
			{
				break;
			}

			/* write out byte and put it in sliding window */
			putc(c, fpOut);
			slidingWindow[nextChar] = c;
			nextChar = Wrap((nextChar + 1), WINDOW_SIZE);
		}
		else
		{
			/* offset and length */
			code.offset = 0;
			code.length = 0;

			if ((fpIn->GetBitsNum(&code.offset, OFFSET_BITS,
								  sizeof(unsigned int))) == EOF)
			// if ((BitFileGetBitsNum(bfpIn, &code.offset, OFFSET_BITS,
			// 					   sizeof(unsigned int))) == EOF)

			{
				break;
			}

			if ((fpIn->GetBitsNum(&code.length, LENGTH_BITS,
								  sizeof(unsigned int))) == EOF)
			// if ((BitFileGetBitsNum(bfpIn, &code.length, LENGTH_BITS,
			// 					   sizeof(unsigned int))) == EOF)
			{
				break;
			}

			code.length += MAX_UNCODED + 1;

			/****************************************************************
			* Write out decoded string to file and lookahead.  It would be
			* nice to write to the sliding window instead of the lookahead,
			* but we could end up overwriting the matching string with the
			* new string if abs(offset - next char) < match length.
			****************************************************************/
			for (i = 0; i < code.length; i++)
			{
				c = slidingWindow[Wrap((code.offset + i), WINDOW_SIZE)];
				putc(c, fpOut);
				uncodedLookahead[i] = c;
			}

			/* write out decoded string to sliding window */
			for (i = 0; i < code.length; i++)
			{
				slidingWindow[Wrap((nextChar + i), WINDOW_SIZE)] =
					uncodedLookahead[i];
			}

			nextChar = Wrap((nextChar + code.length), WINDOW_SIZE);
		}
	}

	/* we've decoded everything, free bitfile structure */
	// BitFileToFILE(bfpIn);
	fpIn->Flush();
	return 0;
}

int LzssDecodeMemory(unsigned char *input, int sizeInput, unsigned char *output, int sizeOutput, int *outDecompressedSize)
{

	auto bitMemoryRef = MakeBitMemory(input, sizeInput, BM_READ);
	int c;
	unsigned int i, nextChar;
	encoded_string_t code;
	int decompressedSize = 0;

	if (decompressedSize >= sizeOutput)
	{
		return SMALL_BUFFER_FAILURE;
	}
	/************************************************************************
	* Fill the sliding window buffer with some known vales.  EncodeLZSS must
	* use the same values.  If common characters are used, there's an
	* increased chance of matching to the earlier strings.
	************************************************************************/
	memset(slidingWindow, ' ', WINDOW_SIZE * sizeof(unsigned char));

	nextChar = 0;

	while (1)
	{
		// if ((c = BitFileGetBit(bfpIn)) == EOF)
		if ((c = BitMemoryGetBit(bitMemoryRef)) == EOF)
		{
			/* we hit the EOF */
			break;
		}

		if (c == UNCODED)
		{
			/* uncoded character */
			if ((c = BitMemoryGetChar(bitMemoryRef)) == EOF)
			// if ((c = BitFileGetChar(bfpIn)) == EOF)
			{
				break;
			}

			/* write out byte and put it in sliding window */

			if (decompressedSize >= sizeOutput)
			{
				return SMALL_BUFFER_FAILURE;
			}
			output[decompressedSize++] = c;
			slidingWindow[nextChar] = c;
			nextChar = Wrap((nextChar + 1), WINDOW_SIZE);
		}
		else
		{
			/* offset and length */
			code.offset = 0;
			code.length = 0;

			if ((BitMemoryGetBitsNum(bitMemoryRef, &code.offset, OFFSET_BITS,
									 sizeof(unsigned int))) == EOF)
			{
				break;
			}

			if ((BitMemoryGetBitsNum(bitMemoryRef, &code.length, LENGTH_BITS,
									 sizeof(unsigned int))) == EOF)
			{
				break;
			}

			code.length += MAX_UNCODED + 1;

			/****************************************************************
			* Write out decoded string to file and lookahead.  It would be
			* nice to write to the sliding window instead of the lookahead,
			* but we could end up overwriting the matching string with the
			* new string if abs(offset - next char) < match length.
			****************************************************************/
			for (i = 0; i < code.length; i++)
			{
				c = slidingWindow[Wrap((code.offset + i), WINDOW_SIZE)];

				if (decompressedSize >= sizeOutput)
				{
					return SMALL_BUFFER_FAILURE;
				}
				output[decompressedSize++] = c;
				uncodedLookahead[i] = c;
			}

			/* write out decoded string to sliding window */
			for (i = 0; i < code.length; i++)
			{
				slidingWindow[Wrap((nextChar + i), WINDOW_SIZE)] =
					uncodedLookahead[i];
			}

			nextChar = Wrap((nextChar + code.length), WINDOW_SIZE);
		}
	}

	/* we've decoded everything, free bitfile structure */
	int dummySize;
	BitMemoryToArray(bitMemoryRef, &dummySize);

	*outDecompressedSize = decompressedSize;
	return 0;
}

int LzssEncodeMemory(unsigned char *input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize)
{
// printf("LzssEncodeMemory %i\n",sizeInput);
	unsigned char slidingWindow[WINDOW_SIZE];
	unsigned char uncodedLookahead[MAX_CODED];
	int counter = 0;
	int error = 0;
	auto bitMemoryRef = MakeBitMemory(output, sizeOutput, BM_WRITE);
	encoded_string_t matchData;
	int c;
	unsigned int i;
	unsigned int len; /* length of string */

	/* head of sliding window and lookahead */
	unsigned int windowHead, uncodedHead;

	int currentInputPosition = 0;

	windowHead = 0;
	uncodedHead = 0;
#define GET_INPUT_CHAR (currentInputPosition < sizeInput ? input[currentInputPosition++] : EOF)
	/************************************************************************
	* Fill the sliding window buffer with some known vales.  DecodeLZSS must
	* use the same values.  If common characters are used, there's an
	* increased chance of matching to the earlier strings.
	************************************************************************/
	memset(slidingWindow, ' ', WINDOW_SIZE * sizeof(unsigned char));

	/************************************************************************
	* Copy MAX_CODED bytes from the input file into the uncoded lookahead
	* buffer.
	************************************************************************/
	for (len = 0; len < MAX_CODED && (c = GET_INPUT_CHAR) != EOF; len++)
	{
		uncodedLookahead[len] = c;
	}

	if (0 == len)
	{
		return 0; /* inFile was empty */
	}

	/* Look for matching string in sliding window */
	i = InitializeSearchStructures();

	if (0 != i)
	{
		return i; /* InitializeSearchStructures returned an error */
	}
	matchData = FindMatch(slidingWindow, uncodedLookahead, windowHead, uncodedHead, counter);
	counter ++;
	/* now encoded the rest of the file until an EOF is read */
	while (len > 0)
	{
		// printf("THISMETTERS %i %i\n",matchData.offset,matchData.offset);
		if (matchData.length > len)
		{
			/* garbage beyond last data happened to extend match length */
			matchData.length = len;
		}

		if (matchData.length <= MAX_UNCODED)
		{
			/* not long enough match.  write uncoded flag and character */
			error = BitMemoryPutBit(UNCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutChar(uncodedLookahead[uncodedHead], bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			matchData.length = 1; /* set to 1 for 1 byte uncoded */
		}
		else
		{
			unsigned int adjustedLen;

			/* adjust the length of the match so minimun encoded len is 0*/
			adjustedLen = matchData.length - (MAX_UNCODED + 1);

			/* match length > MAX_UNCODED.  Encode as offset and length. */
			error = BitMemoryPutBit(ENCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, &matchData.offset, OFFSET_BITS,
								sizeof(unsigned int));
			if(error == EOF){
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
								sizeof(unsigned int));
			if(error == EOF){
				return SMALL_BUFFER_FAILURE;
			}
		}

		/********************************************************************
		* Replace the matchData.length worth of bytes we've matched in the
		* sliding window with new bytes from the input file.
		********************************************************************/
		i = 0;
		while ((i < matchData.length) && ((c = GET_INPUT_CHAR) != EOF))
		{
			/* add old byte into sliding window and new into lookahead */
			ReplaceChar(slidingWindow, windowHead, uncodedLookahead[uncodedHead]);
			uncodedLookahead[uncodedHead] = c;
			windowHead = Wrap((windowHead + 1), WINDOW_SIZE);
			uncodedHead = Wrap((uncodedHead + 1), MAX_CODED);
			i++;
		}

		/* handle case where we hit EOF before filling lookahead */
		while (i < matchData.length)
		{
			ReplaceChar(slidingWindow, windowHead, uncodedLookahead[uncodedHead]);
			/* nothing to add to lookahead here */
			windowHead = Wrap((windowHead + 1), WINDOW_SIZE);
			uncodedHead = Wrap((uncodedHead + 1), MAX_CODED);
			len--;
			i++;
		}

		/* find match for the remaining characters */

		matchData = FindMatch(slidingWindow, uncodedLookahead, windowHead, uncodedHead, counter);
		counter ++;
		
	}

	/* we've encoded everything, free bitfile structure */
	unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, outCompressedSize);
	if(resultArray == NULL){
		return SMALL_BUFFER_FAILURE;
	}
#undef GET_INPUT_CHAR
	return 0;
}
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
#include <memory>
#include <iostream>
#include <errno.h>
#include "lzlocal.h"
#include "lzss.h"
#include "bitfile.h"
#include "statistics.h"
#include "matchers/matcher_base.h"
#include "file_stream.h"
#include <algorithm>
#include "gpu_util.h"
//#include "md5.h"
/****************************************************************************
*   Function   : EncodeLZSSGpu
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

int LzssBatchSize = BATCH_SIZE;


int EncodeLZSSGpu(FileStream *fpIn, FileStream *fpOut, ExecutionPlan plan, AppStatistics *metrics)
{

	/* convert output file to bitfile */
	//Smart pointer to Matcher
	std::unique_ptr<MatcherBase> matcher;
	switch (plan)
	{
	case GPU_CUDA:
		matcher.reset(new MatcherCuda);
		break;
	case GPU_OPENACC:
		matcher.reset(new MatcherOpenAcc);
		break;

	case GPU_OPENCL:
		matcher.reset(new MatcherOpenCl);
		break;
	case CPU_SEQUENTIAL:
	default:
		matcher.reset(new MatcherSequential);
		break;
	}
	matcher->Init();
	int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	//Read the batch size
	int *matches_offset = new int[LzssBatchSize];
	int *matches_length = new int[LzssBatchSize];

	//encoded_string_t matches[batchSize];
	char *buffer = new char[batchSizeWithWindow];
	int c = ' ';
	int current = 0;
	int next_jump = 0;
	int currentMatchCount = 0;
	while (true)
	{
		int i;
		if (current == 0)
		{
			memset(buffer, ' ', (WINDOW_SIZE) * sizeof(char));
		}
		else
		{
			//Recopy last bits not searched in last buffer
			for (int i = 0; i < MAX_CODED; i++)
			{
				buffer[i + WINDOW_SIZE] = buffer[batchSizeWithWindow - MAX_CODED + i];
			}
			for (int i = 0; i < WINDOW_SIZE; i++)
			{
				buffer[i] = buffer[LzssBatchSize + i];
			}
		}
		for (i = WINDOW_SIZE + (current == 0 ? 0 : MAX_CODED); i < batchSizeWithWindow && (c = fpIn->GetChar()) != EOF; i++)
		{
			buffer[i] = c;
		}
		bool isLast = i != batchSizeWithWindow;

		// printDigestMD5(buffer,sizeof(char) * i);

		int matchSize = 0;
		//Find matches in batch
		metrics->StartFindMatch();
		matcher->FindMatchBatch(buffer, i, matches_length, matches_offset, &matchSize, isLast, currentMatchCount, current);
		metrics->StopFindMatch();

		currentMatchCount += i - MAX_CODED;
		int x;
		for (x = next_jump; x < matchSize; x++)
		{

#ifdef PRINTOFFSETS
			printf("%i %i\n", matches_offset[x], matches_length[x]);
#endif
			//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				/* not long enough match.  write uncoded flag and character */
				fpOut->PutBit(UNCODED);
				fpOut->PutChar(buffer[x + WINDOW_SIZE]);
			}
			else
			{
				unsigned int adjustedLen;

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				fpOut->PutBit(ENCODED);
				fpOut->PutBitsNum(matches_offset + x, OFFSET_BITS,
								  sizeof(unsigned int));
				fpOut->PutBitsNum(&adjustedLen, LENGTH_BITS,
								  sizeof(unsigned int));
			}
			// #endif
			if (matches_length[x] > 2)
			{
				x += matches_length[x] - 1;
			}
		}

		next_jump = x - matchSize;

		if (isLast)
		{
			break;
		}
		current++;
	}

	// BitFileToFILE(bfpOut);
	fpOut->Flush();
	delete matches_offset;
	delete matches_length;
#define MICRO_TO_SECONDS(v) v / 1000

	std::cout
		<< "timeSpentOnMemoryHostToDevice: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnMemoryHostToDevice)
		<< " seconds"
		<< std::endl;
	std::cout
		<< "timeSpentOnMemoryDeviceToHost: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnMemoryDeviceToHost)
		<< " seconds"
		<< std::endl;
	std::cout
		<< "timeSpentOnKernel: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnKernel)
		<< " seconds"
		<< std::endl;
#undef MICRO_TO_SECONDS
	return 0;
}

int EncodeLZSSCpuGpu(FileStream *fpIn, FileStream *fpOut, ExecutionPlan plan, AppStatistics *metrics)
{

	/* convert output file to bitfile */
	//Smart pointer to Matcher
	std::unique_ptr<MatcherBase> matcher;
	switch (plan)
	{
	case GPU_CUDA:
		matcher.reset(new MatcherCuda);
		break;
	case GPU_OPENACC:
		matcher.reset(new MatcherOpenAcc);
		break;

	case GPU_OPENCL:
		matcher.reset(new MatcherOpenCl);
		break;
	case CPU_SEQUENTIAL:
	default:
		matcher.reset(new MatcherSequential);
		break;
	}
	matcher->Init();
	int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	//Read the batch size
	int *matches_offset = new int[LzssBatchSize];
	int *matches_length = new int[LzssBatchSize];

	//encoded_string_t matches[LzssBatchSize];
	int c = ' ';
	int current = 0;
	int next_jump = 0;
	int currentMatchCount = 0;
	char *lastBuffer = new char[WINDOW_SIZE + MAX_CODED];
	while (true)
	{
		char *buffer = new char[batchSizeWithWindow];
		if (current == 0)
		{
			memset(buffer, ' ', (WINDOW_SIZE) * sizeof(char));
		}
		else
		{
			memcpy(buffer, lastBuffer, sizeof(char) * (WINDOW_SIZE + MAX_CODED));
			//Recopy last bits not searched in last buffer
			// for (int i = 0; i < MAX_CODED; i++) {
			// 	buffer[i + WINDOW_SIZE] = lastBuffer[WINDOW_SIZE + i];
			// }
			// for (int i = 0; i < WINDOW_SIZE; i++) {
			// 	buffer[i] = lastBuffer[i];
			// }
		}
		int i;
		for (i = WINDOW_SIZE + (current == 0 ? 0 : MAX_CODED); i < batchSizeWithWindow && (c = fpIn->GetChar()) != EOF; i++)
		{
			buffer[i] = c;
		}
		bool isLast = i != batchSizeWithWindow;
		if (isLast)
		{
			buffer[i] = '\0';
			memset(buffer + i, ' ', (batchSizeWithWindow - i) * sizeof(char));
		}
		// printDigestMD5(buffer,sizeof(char) * i);
		// lastBuffer = buffer;
		if (!isLast)
		{
			memcpy(lastBuffer, buffer + (batchSizeWithWindow - WINDOW_SIZE - MAX_CODED), sizeof(char) * (MAX_CODED + WINDOW_SIZE));
		}
		// std::copy_n(buffer +(batchSizeWithWindow - WINDOW_SIZE - MAX_CODED),WINDOW_SIZE + MAX_CODED,lastBuffer);
		int matchSize = 0;
		//Find matches in batch
		metrics->StartFindMatch();
		matcher->FindMatchBatch(buffer, i, matches_length, matches_offset, &matchSize, isLast, currentMatchCount, current);
		metrics->StopFindMatch();

		currentMatchCount += i - MAX_CODED;
		int x;
		for (x = next_jump; x < matchSize; x++)
		{

#ifdef PRINTOFFSETS
			printf("%i %i\n", matches_offset[x], matches_length[x]);
#endif
			//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				/* not long enough match.  write uncoded flag and character */
				fpOut->PutBit(UNCODED);
				fpOut->PutChar(buffer[x + WINDOW_SIZE]);
			}
			else
			{
				unsigned int adjustedLen;

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				fpOut->PutBit(ENCODED);
				fpOut->PutBitsNum(matches_offset + x, OFFSET_BITS,
								  sizeof(unsigned int));
				fpOut->PutBitsNum(&adjustedLen, LENGTH_BITS,
								  sizeof(unsigned int));
			}
			// #endif
			if (matches_length[x] > 2)
			{
				x += matches_length[x] - 1;
			}
		}
		delete[] buffer;

		next_jump = x - matchSize;

		if (isLast)
		{
			break;
		}
		current++;
	}

	// BitFileToFILE(bfpOut);
	fpOut->Flush();
	delete[] matches_offset;
	delete[] matches_length;
	delete[] lastBuffer;
#define MICRO_TO_SECONDS(v) v / 1000

	std::cout
		<< "timeSpentOnMemoryHostToDevice: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnMemoryHostToDevice)
		<< " seconds"
		<< std::endl;
	std::cout
		<< "timeSpentOnMemoryDeviceToHost: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnMemoryDeviceToHost)
		<< " seconds"
		<< std::endl;
	std::cout
		<< "timeSpentOnKernel: "
		<< MICRO_TO_SECONDS(matcher->timeSpentOnKernel)
		<< " seconds"
		<< std::endl;
#undef MICRO_TO_SECONDS
	return 0;
}

int LzssEncodeMemoryGpu(unsigned char *input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize)
{
	/* convert output file to bitfile */
	//Smart pointer to Matcher
	std::unique_ptr<MatcherBase> matcher;

	ExecutionPlan plan = GPU_CUDA;
	switch (plan)
	{
	case GPU_CUDA:
		matcher.reset(new MatcherCuda);
		break;
	case GPU_OPENACC:
		matcher.reset(new MatcherOpenAcc);
		break;

	case GPU_OPENCL:
		matcher.reset(new MatcherOpenCl);
		break;
	case CPU_SEQUENTIAL:
	default:
		matcher.reset(new MatcherSequential);
		break;
	}

	auto bitMemoryRef = MakeBitMemory(output, sizeOutput, BM_WRITE);

	matcher->Init();
	int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	int error = 0;
	//Read the batch size
	int *matches_offset = new int[LzssBatchSize];
	int *matches_length = new int[LzssBatchSize];
	int currentInputPosition = 0;
#define GET_INPUT_CHAR (currentInputPosition < sizeInput ? input[currentInputPosition++] : EOF)

	//encoded_string_t matches[LzssBatchSize];
	char *buffer = new char[batchSizeWithWindow];
	int c = ' ';
	int current = 0;
	int next_jump = 0;
	int currentMatchCount = 0;
	while (true)
	{
		int i;
		if (current == 0)
		{
			memset(buffer, ' ', (WINDOW_SIZE) * sizeof(char));
		}
		else
		{
			//Recopy last bits not searched in last buffer
			for (int i = 0; i < MAX_CODED; i++)
			{
				buffer[i + WINDOW_SIZE] = buffer[batchSizeWithWindow - MAX_CODED + i];
			}
			for (int i = 0; i < WINDOW_SIZE; i++)
			{
				buffer[i] = buffer[LzssBatchSize + i];
			}
		}
		for (i = WINDOW_SIZE + (current == 0 ? 0 : MAX_CODED); i < batchSizeWithWindow && (c = GET_INPUT_CHAR) != EOF; i++)
		{
			buffer[i] = c;
		}
		bool isLast = i != batchSizeWithWindow;

		// printDigestMD5(buffer,sizeof(char) * i);

		int matchSize = 0;
		//Find matches in batch
		matcher->FindMatchBatch(buffer, i, matches_length, matches_offset, &matchSize, isLast, currentMatchCount, current);

		currentMatchCount += i - MAX_CODED;
		int x;
		for (x = next_jump; x < matchSize; x++)
		{

#ifdef PRINTOFFSETS
			printf("%i %i\n", matches_offset[x], matches_length[x]);
#endif
			//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				/* not long enough match.  write uncoded flag and character */
				error = BitMemoryPutBit(UNCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutChar(buffer[x + WINDOW_SIZE], bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
			else
			{
				unsigned int adjustedLen;

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				error = BitMemoryPutBit(ENCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
			// #endif
			if (matches_length[x] > 2)
			{
				x += matches_length[x] - 1;
			}
		}

		next_jump = x - matchSize;

		if (isLast)
		{
			break;
		}
		current++;
	}
#undef GET_INPUT_CHAR
	unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, outCompressedSize);
	if(resultArray == NULL){
		return SMALL_BUFFER_FAILURE;
	}
	delete matches_offset;
	delete matches_length;
	return 0;
}


int LzssEncodeMemoryGpu(unsigned char *input, unsigned char *d_input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize, int deviceIndex)
{

	/* convert output file to bitfile */
	//Smart pointer to Matcher
	std::unique_ptr<MatcherBase> matcher;

	ExecutionPlan plan = GPU_CUDA;
	// switch (plan)
	// {
	// case GPU_CUDA:
		matcher.reset(new MatcherCuda);
	// 	break;
	// case GPU_OPENACC:
	// 	matcher.reset(new MatcherOpenAcc);
	// 	break;

	// case GPU_OPENCL:
	// 	matcher.reset(new MatcherOpenCl);
	// 	break;
	// case CPU_SEQUENTIAL:
	// default:
	// 	matcher.reset(new MatcherSequential);
	// 	break;
	// }

	auto bitMemoryRef = MakeBitMemory(output, sizeOutput, BM_WRITE);

	matcher->Init();
	int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	//Read the batch size
	int *matches_offset = new int[LzssBatchSize];
	int *matches_length = new int[LzssBatchSize];

	//encoded_string_t matches[LzssBatchSize];
	int error = 0;
	int matchSize = 0;
	//Find matches in batch
	error = matcher->FindMatchBatchUsingDeviceInput(d_input, sizeInput, matches_length, matches_offset, &matchSize,deviceIndex);
	if (error != 0)
	{
		return error;
	}
	int x;
	for (x = 0; x < matchSize; x++)
	{
		//Save to file
		if (matches_length[x] <= MAX_UNCODED)
		{
			/* not long enough match.  write uncoded flag and character */
			error = BitMemoryPutBit(UNCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutChar(input[x], bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
		}
		else
		{
			unsigned int adjustedLen;

			/* adjust the length of the match so minimun encoded len is 0*/
			adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

			/* match length > MAX_UNCODED.  Encode as offset and length. */
			error = BitMemoryPutBit(ENCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
										sizeof(unsigned int));
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
										sizeof(unsigned int));
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
		}
		// #endif
		if (matches_length[x] > 2)
		{
			x += matches_length[x] - 1;
		}
	}

	unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, outCompressedSize);
	if(resultArray == NULL){
		return SMALL_BUFFER_FAILURE;
	}
	delete matches_offset;
	delete matches_length;
	return 0;
}

int LzssEncodeMemoryGpu(unsigned char *input, unsigned char *d_input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize)
{

	/* convert output file to bitfile */
	//Smart pointer to Matcher
	std::unique_ptr<MatcherBase> matcher;

	ExecutionPlan plan = GPU_CUDA;
	// switch (plan)
	// {
	// case GPU_CUDA:
		matcher.reset(new MatcherCuda);
	// 	break;
	// case GPU_OPENACC:
	// 	matcher.reset(new MatcherOpenAcc);
	// 	break;

	// case GPU_OPENCL:
	// 	matcher.reset(new MatcherOpenCl);
	// 	break;
	// case CPU_SEQUENTIAL:
	// default:
	// 	matcher.reset(new MatcherSequential);
	// 	break;
	// }

	auto bitMemoryRef = MakeBitMemory(output, sizeOutput, BM_WRITE);

	matcher->Init();
	int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	//Read the batch size
	int *matches_offset = new int[LzssBatchSize];
	int *matches_length = new int[LzssBatchSize];

	//encoded_string_t matches[LzssBatchSize];
	int error = 0;
	int matchSize = 0;
	//Find matches in batch
	error = matcher->FindMatchBatchUsingDeviceInput(d_input, sizeInput, matches_length, matches_offset, &matchSize,0);
	if (error != 0)
	{
		return error;
	}
	int x;
	for (x = 0; x < matchSize; x++)
	{
		//Save to file
		if (matches_length[x] <= MAX_UNCODED)
		{
			/* not long enough match.  write uncoded flag and character */
			error = BitMemoryPutBit(UNCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutChar(input[x], bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
		}
		else
		{
			unsigned int adjustedLen;

			/* adjust the length of the match so minimun encoded len is 0*/
			adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

			/* match length > MAX_UNCODED.  Encode as offset and length. */
			error = BitMemoryPutBit(ENCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
										sizeof(unsigned int));
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
										sizeof(unsigned int));
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
		}
		// #endif
		if (matches_length[x] > 2)
		{
			x += matches_length[x] - 1;
		}
	}

	unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, outCompressedSize);
	if(resultArray == NULL){
		return SMALL_BUFFER_FAILURE;
	}
	delete matches_offset;
	delete matches_length;
	return 0;
}




int LzssEncodeMemoryGpu(unsigned char *input, cl::Buffer d_input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize)
{

	/* convert output file to bitfile */
	//Smart pointer to Matcher
	std::unique_ptr<MatcherBase> matcher;

	ExecutionPlan plan = GPU_OPENCL;
	matcher.reset(new MatcherOpenCl);
	// switch (plan)
	// {
	// case GPU_CUDA:
	// 	break;
	// case GPU_OPENACC:
	// 	matcher.reset(new MatcherOpenAcc);
	// 	break;

	// case GPU_OPENCL:
	// 	matcher.reset(new MatcherOpenCl);
	// 	break;
	// case CPU_SEQUENTIAL:
	// default:
	// 	matcher.reset(new MatcherSequential);
	// 	break;
	// }

	auto bitMemoryRef = MakeBitMemory(output, sizeOutput, BM_WRITE);

	matcher->Init();
	int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	//Read the batch size
	int *matches_offset = new int[LzssBatchSize];
	int *matches_length = new int[LzssBatchSize];

	//encoded_string_t matches[LzssBatchSize];
	int error = 0;
	int matchSize = 0;
	//Find matches in batch
	error = matcher->FindMatchBatchUsingDeviceInput(d_input, sizeInput,0, matches_length, matches_offset, &matchSize);
	if (error != 0)
	{
		return error;
	}
	int x;
	for (x = 0; x < matchSize; x++)
	{
		//Save to file
		if (matches_length[x] <= MAX_UNCODED)
		{
			/* not long enough match.  write uncoded flag and character */
			error = BitMemoryPutBit(UNCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutChar(input[x], bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
		}
		else
		{
			unsigned int adjustedLen;

			/* adjust the length of the match so minimun encoded len is 0*/
			adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

			/* match length > MAX_UNCODED.  Encode as offset and length. */
			error = BitMemoryPutBit(ENCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
										sizeof(unsigned int));
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
										sizeof(unsigned int));
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
		}
		// #endif
		if (matches_length[x] > 2)
		{
			x += matches_length[x] - 1;
		}
	}

	unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, outCompressedSize);
	if(resultArray == NULL){
		return SMALL_BUFFER_FAILURE;
	}
	delete matches_offset;
	delete matches_length;
	return 0;
}




int LzssEncodeMemoryGpu(unsigned char *input, cl::Buffer d_input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize, int deviceId)
{

	/* convert output file to bitfile */
	//Smart pointer to Matcher
	std::unique_ptr<MatcherOpenCl> matcher;

	ExecutionPlan plan = GPU_OPENCL;
	matcher.reset(new MatcherOpenCl);
	// switch (plan)
	// {
	// case GPU_CUDA:
	// 	break;
	// case GPU_OPENACC:
	// 	matcher.reset(new MatcherOpenAcc);
	// 	break;

	// case GPU_OPENCL:
	// 	matcher.reset(new MatcherOpenCl);
	// 	break;
	// case CPU_SEQUENTIAL:
	// default:
	// 	matcher.reset(new MatcherSequential);
	// 	break;
	// }

	auto bitMemoryRef = MakeBitMemory(output, sizeOutput, BM_WRITE);

	matcher->Init();
	int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	//Read the batch size
	int *matches_offset = new int[LzssBatchSize];
	int *matches_length = new int[LzssBatchSize];

	//encoded_string_t matches[LzssBatchSize];
	int error = 0;
	int matchSize = 0;
	//Find matches in batch
	error = matcher->FindMatchBatchUsingDeviceInput(d_input, sizeInput,0, matches_length, matches_offset, &matchSize,deviceId);
	if (error != 0)
	{
		return error;
	}
	int x;
	for (x = 0; x < matchSize; x++)
	{
		//Save to file
		if (matches_length[x] <= MAX_UNCODED)
		{
			/* not long enough match.  write uncoded flag and character */
			error = BitMemoryPutBit(UNCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutChar(input[x], bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
		}
		else
		{
			unsigned int adjustedLen;

			/* adjust the length of the match so minimun encoded len is 0*/
			adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

			/* match length > MAX_UNCODED.  Encode as offset and length. */
			error = BitMemoryPutBit(ENCODED, bitMemoryRef);
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
										sizeof(unsigned int));
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
			error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
										sizeof(unsigned int));
			if (error == EOF)
			{
				return SMALL_BUFFER_FAILURE;
			}
		}
		// #endif
		if (matches_length[x] > 2)
		{
			x += matches_length[x] - 1;
		}
	}

	unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, outCompressedSize);
	if(resultArray == NULL){
		return SMALL_BUFFER_FAILURE;
	}
	delete matches_offset;
	delete matches_length;
	return 0;
}



int LzssEncodeMemoryCpuBatch(unsigned char *input, int sizeInput, int * breakPositions, int breakSize, std::vector<std::vector<unsigned char>> *output){
	/* convert output file to bitfile */
	//Smart pointer to Matcher
	auto matcher = new MatcherSequential();
	int sizeOutput = sizeInput* 2;
	int matchSize = sizeInput;
	// ExecutionPlan plan = GPU_CUDA;
	// switch (plan)
	// {
	// case GPU_CUDA:
	// 	break;
	// case GPU_OPENACC:
	// 	matcher.reset(new MatcherOpenAcc);
	// 	break;

	// case GPU_OPENCL:
	// 	matcher.reset(new MatcherOpenCl);
	// 	break;
	// case CPU_SEQUENTIAL:
	// default:
	// 	matcher.reset(new MatcherSequential);
	// 	break;
	// }
	unsigned char * outputArray = new unsigned char[sizeOutput];


	// matcher->Init();
	// int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	// //Read the batch size
	int *matches_offset = new int[sizeInput];
	int *matches_length = new int[sizeInput];



	matcher->FindMatchBatchInMemory(input,sizeInput,breakPositions,breakSize,matches_length,matches_offset);
	// //encoded_string_t matches[LzssBatchSize];
	int error = 0;
	// int matchSize = 0;
	// //Find matches in batch
	// error = matcher->FindMatchBatch(d_input, sizeInput, matches_length, matches_offset, &matchSize);
	// if (error != 0)
	// {
	// 	return error;
	// }
	
	// printf("matchSize: %i\n",matchSize);
	auto original_input = input;
	auto original_matches_length = matches_length;
	auto original_matches_offset = matches_offset;
	for(int i = 0; i < breakSize; i++){
		auto startPos = breakPositions[i];
		input = original_input + startPos;
		matches_offset = original_matches_offset + startPos;
		matches_length = original_matches_length + startPos;
		auto lastPos = (i + 1) == breakSize? sizeInput:breakPositions[i+1];
		matchSize = lastPos - startPos;
		int x;
		auto bitMemoryRef = MakeBitMemory(outputArray, sizeOutput, BM_WRITE);
				// printf("\n\n\n\n new \n\n\n\n\n ");

		for (x = 0; x < matchSize; x++)
		{
			// printf("%c ",input[x]);
		// 	//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				// printf("UNCODED(%c) ",input[x]);
				
				/* not long enough match.  write uncoded flag and character */
				error = BitMemoryPutBit(UNCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutChar(input[x], bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
			else
			{
				unsigned int adjustedLen;
				int offset = matches_offset[x];
				int length = matches_length[x];
				
				// printf("ENCODED(%i/%i) ",offset,length);

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = length - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				error = BitMemoryPutBit(ENCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
		// 	// #endif
			if (matches_length[x] > 2)
			{
				// printf("JUMP(%i) ", matches_length[x] -1);
				x += matches_length[x] - 1;
				// printf("current x %i ",x);
			}
		}
		int newSize =0;
		unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, &newSize);
		if(resultArray == NULL){
			return SMALL_BUFFER_FAILURE;
		}
		std::vector<unsigned char> vec(resultArray, resultArray + newSize);
		output->push_back(vec);
	}

	
	delete original_matches_offset;
	delete original_matches_length;
	return 0;
	// return -1;
}




int LzssEncodeMemoryGpuBatch(unsigned char *input,unsigned char *d_input, int sizeInput, int * breakPositions, int * d_breakPositions, int breakSize, std::vector<std::vector<unsigned char>> *output,int device){
	/* convert output file to bitfile */
	//Smart pointer to Matcher
	auto matcher = new MatcherCuda();
	int sizeOutput = sizeInput* 2;
	int matchSize = sizeInput;
	
	unsigned char * outputArray = new unsigned char[sizeOutput];


	// matcher->Init();
	// int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	// //Read the batch size
	int *matches_offset = new int[sizeInput];
	int *matches_length = new int[sizeInput];



	matcher->FindMatchBatchInMemory(d_input,sizeInput,d_breakPositions,breakSize,matches_length,matches_offset,device);
	// //encoded_string_t matches[LzssBatchSize];
	int error = 0;
	// int matchSize = 0;
	// //Find matches in batch
	// error = matcher->FindMatchBatch(d_input, sizeInput, matches_length, matches_offset, &matchSize);
	// if (error != 0)
	// {
	// 	return error;
	// }
	
	// printf("matchSize: %i\n",matchSize);
	auto original_input = input;
	auto original_matches_length = matches_length;
	auto original_matches_offset = matches_offset;
	for(int i = 0; i < breakSize; i++){
		auto startPos = breakPositions[i];
		input = original_input + startPos;
		matches_offset = original_matches_offset + startPos;
		matches_length = original_matches_length + startPos;
		auto lastPos = (i + 1) == breakSize? sizeInput:breakPositions[i+1];
		matchSize = lastPos - startPos;
		int x;
		auto bitMemoryRef = MakeBitMemory(outputArray, sizeOutput, BM_WRITE);

		for (x = 0; x < matchSize; x++)
		{
			// printf("%c ",input[x]);
		// 	//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				// printf("UNCODED(%c) ",input[x]);
				
				/* not long enough match.  write uncoded flag and character */
				error = BitMemoryPutBit(UNCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutChar(input[x], bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
			else
			{
				unsigned int adjustedLen;
				// printf("ENCODED(%i/%i) ",matches_offset[x],matches_length[x]);

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				error = BitMemoryPutBit(ENCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
		// 	// #endif
			if (matches_length[x] > 2)
			{
				// printf("JUMP(%i) ", matches_length[x] -1);
				x += matches_length[x] - 1;
			}
		}
		int newSize =0;
		unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, &newSize);
		if(resultArray == NULL){
			return SMALL_BUFFER_FAILURE;
		}
		std::vector<unsigned char> vec(resultArray, resultArray + newSize);
		output->push_back(vec);
	}

	
	delete original_matches_offset;
	delete original_matches_length;
	return 0;
	// return -1;
}

int LzssEncodeMemoryGpuBatch(unsigned char *input,unsigned char *d_input, int sizeInput, int * breakPositions, int * d_breakPositions, int breakSize, std::vector<std::vector<unsigned char>> *output,int device,cudaStream_t cuda_stream){
	/* convert output file to bitfile */
	//Smart pointer to Matcher
	auto matcher = new MatcherCuda();
	int sizeOutput = sizeInput* 2;
	int matchSize = sizeInput;
	
	unsigned char * outputArray = new unsigned char[sizeOutput];


	// matcher->Init();
	// int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	// //Read the batch size
	int *matches_offset;
	int *matches_length;
	int err;
	err = cudaHostAlloc((void**)&matches_offset, sizeof(int) * sizeInput, cudaHostAllocDefault);
	if(err != cudaSuccess){
		fprintf(stderr, "Error: %s %d\n",  __FILE__, __LINE__);
		exit(err);
	}
	err = cudaHostAlloc((void**)&matches_length, sizeof(int) * sizeInput, cudaHostAllocDefault);
	if(err != cudaSuccess){
		fprintf(stderr, "Error: %s %d\n",   __FILE__, __LINE__);
		exit(err);
	}




	matcher->FindMatchBatchInMemory(d_input,sizeInput,d_breakPositions,breakSize,matches_length,matches_offset,device,cuda_stream);
	// //encoded_string_t matches[LzssBatchSize];
	int error = 0;
	// int matchSize = 0;
	// //Find matches in batch
	// error = matcher->FindMatchBatch(d_input, sizeInput, matches_length, matches_offset, &matchSize);
	// if (error != 0)
	// {
	// 	return error;
	// }
	
	// printf("matchSize: %i\n",matchSize);
	auto original_input = input;
	auto original_matches_length = matches_length;
	auto original_matches_offset = matches_offset;
	for(int i = 0; i < breakSize; i++){
		auto startPos = breakPositions[i];
		input = original_input + startPos;
		matches_offset = original_matches_offset + startPos;
		matches_length = original_matches_length + startPos;
		auto lastPos = (i + 1) == breakSize? sizeInput:breakPositions[i+1];
		matchSize = lastPos - startPos;
		int x;
		auto bitMemoryRef = MakeBitMemory(outputArray, sizeOutput, BM_WRITE);

		for (x = 0; x < matchSize; x++)
		{
			// printf("%c ",input[x]);
		// 	//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				// printf("UNCODED(%c) ",input[x]);
				
				/* not long enough match.  write uncoded flag and character */
				error = BitMemoryPutBit(UNCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutChar(input[x], bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
			else
			{
				unsigned int adjustedLen;
				// printf("ENCODED(%i/%i) ",matches_offset[x],matches_length[x]);

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				error = BitMemoryPutBit(ENCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
		// 	// #endif
			if (matches_length[x] > 2)
			{
				// printf("JUMP(%i) ", matches_length[x] -1);
				x += matches_length[x] - 1;
			}
		}
		int newSize =0;
		unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, &newSize);
		if(resultArray == NULL){
			return SMALL_BUFFER_FAILURE;
		}
		std::vector<unsigned char> vec(resultArray, resultArray + newSize);
		output->push_back(vec);
	}

	
	cudaFreeHost(original_matches_offset);
	cudaFreeHost(original_matches_length);
	return 0;
	// return -1;
}




#ifndef OPENCL_C

int LzssEncodeMemoryGpuBatch(unsigned char *input,cl::Buffer d_input, int sizeInput, int * breakPositions, cl::Buffer d_breakPositions, int breakSize, std::vector<std::vector<unsigned char>> *output,int device){
	/* convert output file to bitfile */
	//Smart pointer to Matcher
	auto matcher = new MatcherOpenCl();
	matcher->Init();
	int sizeOutput = sizeInput* 2;
	int matchSize = sizeInput;
	
	unsigned char * outputArray = new unsigned char[sizeOutput];


	// matcher->Init();
	// int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	// //Read the batch size
	int *matches_offset = new int[sizeInput];
	int *matches_length = new int[sizeInput];



	matcher->FindMatchBatchInMemory(d_input,sizeInput,d_breakPositions,breakSize,matches_length,matches_offset,device);
	// //encoded_string_t matches[LzssBatchSize];
	int error = 0;
	// int matchSize = 0;
	// //Find matches in batch
	// error = matcher->FindMatchBatch(d_input, sizeInput, matches_length, matches_offset, &matchSize);
	// if (error != 0)
	// {
	// 	return error;
	// }
	
	// printf("matchSize: %i\n",matchSize);
	auto original_input = input;
	auto original_matches_length = matches_length;
	auto original_matches_offset = matches_offset;
	for(int i = 0; i < breakSize; i++){
		auto startPos = breakPositions[i];
		input = original_input + startPos;
		matches_offset = original_matches_offset + startPos;
		matches_length = original_matches_length + startPos;
		auto lastPos = (i + 1) == breakSize? sizeInput:breakPositions[i+1];
		matchSize = lastPos - startPos;
		int x;
		auto bitMemoryRef = MakeBitMemory(outputArray, sizeOutput, BM_WRITE);

		for (x = 0; x < matchSize; x++)
		{
			// printf("%c ",input[x]);
		// 	//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				// printf("UNCODED(%c) ",input[x]);
				
				/* not long enough match.  write uncoded flag and character */
				error = BitMemoryPutBit(UNCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutChar(input[x], bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
			else
			{
				unsigned int adjustedLen;
				// printf("ENCODED(%i/%i) ",matches_offset[x],matches_length[x]);

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				error = BitMemoryPutBit(ENCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
		// 	// #endif
			if (matches_length[x] > 2)
			{
				// printf("JUMP(%i) ", matches_length[x] -1);
				x += matches_length[x] - 1;
			}
		}
		int newSize =0;
		unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, &newSize);
		if(resultArray == NULL){
			return SMALL_BUFFER_FAILURE;
		}
		std::vector<unsigned char> vec(resultArray, resultArray + newSize);
		output->push_back(vec);
	}

	
	delete original_matches_offset;
	delete original_matches_length;
	return 0;
	// return -1;
}
#else

int LzssEncodeMemoryGpuBatch(unsigned char *input,cl_mem d_input, size_t sizeInput, int * breakPositions, cl_mem d_breakPositions, size_t breakSize, std::vector<std::vector<unsigned char>> *output, OCLDeviceObjects* device, cl_command_queue queue){
	/* convert output file to bitfile */
	//Smart pointer to Matcher
	auto matcher = new MatcherOpenCl();
	// matcher->Init(device);
	int sizeOutput = sizeInput* 2;
	int matchSize = sizeInput;
	
	unsigned char * outputArray = new unsigned char[sizeOutput];


	// matcher->Init();
	// int batchSizeWithWindow = LzssBatchSize + WINDOW_SIZE + MAX_CODED;
	// //Read the batch size
	int *matches_offset = new int[sizeInput];
	int *matches_length = new int[sizeInput];



	matcher->FindMatchBatchInMemory(d_input,sizeInput,d_breakPositions,breakSize,matches_length,matches_offset,device,queue);
	// //encoded_string_t matches[LzssBatchSize];
	int error = 0;
	// int matchSize = 0;
	// //Find matches in batch
	// error = matcher->FindMatchBatch(d_input, sizeInput, matches_length, matches_offset, &matchSize);
	// if (error != 0)
	// {
	// 	return error;
	// }
	
	// printf("matchSize: %i\n",matchSize);
	auto original_input = input;
	auto original_matches_length = matches_length;
	auto original_matches_offset = matches_offset;
	for(int i = 0; i < breakSize; i++){
		auto startPos = breakPositions[i];
		input = original_input + startPos;
		matches_offset = original_matches_offset + startPos;
		matches_length = original_matches_length + startPos;
		auto lastPos = (i + 1) == breakSize? sizeInput:breakPositions[i+1];
		matchSize = lastPos - startPos;
		int x;
		auto bitMemoryRef = MakeBitMemory(outputArray, sizeOutput, BM_WRITE);

		for (x = 0; x < matchSize; x++)
		{
			// printf("%c ",input[x]);
		// 	//Save to file
			if (matches_length[x] <= MAX_UNCODED)
			{
				// printf("UNCODED(%c) ",input[x]);
				
				/* not long enough match.  write uncoded flag and character */
				error = BitMemoryPutBit(UNCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutChar(input[x], bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
			else
			{
				unsigned int adjustedLen;
				// printf("ENCODED(%i/%i) ",matches_offset[x],matches_length[x]);

				/* adjust the length of the match so minimun encoded len is 0*/
				adjustedLen = matches_length[x] - (MAX_UNCODED + 1);

				/* match length > MAX_UNCODED.  Encode as offset and length. */
				error = BitMemoryPutBit(ENCODED, bitMemoryRef);
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, matches_offset + x, OFFSET_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
				error = BitMemoryPutBitsNum(bitMemoryRef, &adjustedLen, LENGTH_BITS,
											sizeof(unsigned int));
				if (error == EOF)
				{
					return SMALL_BUFFER_FAILURE;
				}
			}
		// 	// #endif
			if (matches_length[x] > 2)
			{
				// printf("JUMP(%i) ", matches_length[x] -1);
				x += matches_length[x] - 1;
			}
		}
		int newSize =0;
		unsigned char * resultArray = BitMemoryToArray(bitMemoryRef, &newSize);
		if(resultArray == NULL){
			return SMALL_BUFFER_FAILURE;
		}
		std::vector<unsigned char> vec(resultArray, resultArray + newSize);
		output->push_back(vec);
	}

	
	delete original_matches_offset;
	delete original_matches_length;
	return 0;
	// return -1;
}
#endif
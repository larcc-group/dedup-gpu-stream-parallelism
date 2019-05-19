#include "matcher_base.h"
#include "lzlocal.h"
#include <iostream>

int MatcherSequential::Init()
{
	MatcherBase::Init();
	return 0;
}
int MatcherSequential::FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch)
{

	int bufferSizeAdjusted = bufferSize - MAX_CODED;
	if (isLast)
	{
		bufferSizeAdjusted += MAX_CODED;
	}
	int matchCount = bufferSizeAdjusted - WINDOW_SIZE;
	*matchSize = matchCount;
	for (int idX = 0; idX < matchCount; idX++)
	{
		int i = WINDOW_SIZE + idX;
		int beginSearch = idX;
		

		int length = 0;
		int offset = 0;
		int windowHead = (currentMatchCount + idX) % WINDOW_SIZE;

		int currentOffset = 0;

		// char* current = buffer;
		int j = 0;
		while (1)
		{
			if (buffer[i + 0] == buffer[beginSearch + Wrap((currentOffset), WINDOW_SIZE)])
			{
				/* we matched one. how many more match? */
				j = 1;

				while (
					buffer[i + j] == buffer[beginSearch + Wrap((currentOffset + j), WINDOW_SIZE)] && (!isLast ||
																									  (beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < bufferSizeAdjusted && i + j < bufferSizeAdjusted)))
				{

					if (j >= MAX_CODED)
					{
						break;
					}
					j++;
				}

				if (j > length)
				{

					length = j;
					offset = Wrap((currentOffset + windowHead), WINDOW_SIZE);
				}
			}

			if (j >= MAX_CODED)
			{
				length = MAX_CODED;
				break;
			}

			currentOffset++;

			if (currentOffset == WINDOW_SIZE)
			{
				break;
			}
		}
		matches_offset[idX] = offset;
		matches_length[idX] = length;
	}

	return 0;
}


int MatcherSequential::FindMatchBatchInMemory(unsigned char *input, int sizeInput, int * breakPositions, int breakSize,int *matches_length, int *matches_offset)
{

	for (int idX = 0; idX < sizeInput; idX++)
	{

		int i = idX;
		int startPos = 0;
		int lengthPos = 0;
		int lastPos = 0;
		int found_at = 0;
		for(int k = 0; k < breakSize; k++){
			startPos = breakPositions[k] < i + 1 ? breakPositions[k] : startPos;
			found_at = breakPositions[k] < i + 1? k : found_at;
		}
		lastPos =found_at == breakSize - 1 ?sizeInput: breakPositions[found_at+1] ;
		lengthPos =lastPos-startPos ;
		if( i > lengthPos - 10){
			matches_length[i] = 0;
			matches_offset[i] = 0;
			continue;
		}
		
		auto uncodedLookahead = input + i;

		int thisBatchI = i - startPos ;
		int startWindowSize =std::max(thisBatchI - WINDOW_SIZE,0);
		// printf("i:%i lengthPos: %i, startPos:%i lastPos:%i \n",i,lengthPos,startPos,lastPos);

		// printf("i:%i startWindowSize:%i thisBatchI:%i \n",i,startWindowSize,thisBatchI);
		int longest_length = 0;
		int longest_offset = 0;
		int windowHead = thisBatchI % WINDOW_SIZE;

		for(int current = startWindowSize; current < thisBatchI; current++ ){
			if(input[current+ startPos] == uncodedLookahead[0]){
				int j = 1;
				while(lastPos > current+ startPos + j && input[current+ startPos + j] == uncodedLookahead[j]
				//limits the uncoded lookahead
				&& i + j < lastPos
				//find until start of uncodedLookahead	
				&& current + j< thisBatchI
				){
					if (j >= MAX_CODED)
					{
						break;
					}
					j++;
				}
				if(j > longest_length){
					longest_length = j;
					longest_offset = Wrap(current, WINDOW_SIZE);
				}
			}
		}
		// if(longest_length >= 2){
		// 	printf("Found %c at %i/%i \n",input[i],longest_offset,longest_length);
			
		// }
		matches_offset[idX] = longest_offset;
		matches_length[idX] = longest_length;
	}

	return 0;
}

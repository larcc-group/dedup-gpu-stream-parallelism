
#define OFFSET_BITS     12
#define LENGTH_BITS     4

#if (((1 << (OFFSET_BITS + LENGTH_BITS)) - 1) > UINT_MAX)
#error "Size of encoded data must not exceed the size of an unsigned int"
#endif

/* We want a sliding window*/
#define WINDOW_SIZE     (1 << OFFSET_BITS)

/* maximum match length not encoded and maximum length encoded (4 bits) */
#define MAX_UNCODED     2
#define MAX_CODED       ((1 << LENGTH_BITS) + MAX_UNCODED)

#define ENCODED     0       /* encoded string */
#define UNCODED     1       /* unencoded character */

#define Wrap(value, limit) \
    (((value) < (limit)) ? (value) : ((value) - (limit)))

__kernel
void  FindMatchBatchKernel(__global char* buffer, int bufferSize,__global int* matches_length, __global int* matches_offset,int bufferSizeAdjusted, int currentMatchCount,  int isLast) {
    
    int idX = get_global_id(0);//blockIdx.x*blockDim.x+threadIdx.x;
    int i = WINDOW_SIZE + idX;
    int beginSearch = idX;
    if( i >= bufferSizeAdjusted){
        return;
    }

    //Uncoded Lookahead optimization
    char current[MAX_CODED];
    for (int j = 0; j < MAX_CODED; j++)
    {
        current[j] = buffer[i + j];
    }
    
    int length = 0;
    int offset = 0;
    int windowHead = (currentMatchCount + idX) % WINDOW_SIZE;
  
    int currentOffset = 0;

    // char* current = buffer;
    int j = 0;
    while (1) {
        if (current[0] == buffer[beginSearch  + Wrap((currentOffset), WINDOW_SIZE)]) {
            /* we matched one. how many more match? */
            j = 1;
            
            while (
                current[j] == buffer[beginSearch  + Wrap((currentOffset + j),WINDOW_SIZE)]
                && (!isLast ||
                ( beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < bufferSizeAdjusted
                && i + j < bufferSizeAdjusted) )
                ) {
                        
                if (j >= MAX_CODED) {
                    break;
                }					
                j++;
            }
            
            if (j > length) {
                
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
        
        if (currentOffset == WINDOW_SIZE) {
            break;
        }

    }
    matches_offset[idX] = offset;
    matches_length[idX] = length;
}




__kernel void FindMatchBatchKernelWithoutBuffer(__global unsigned char *buffer, int bufferSize,__global int *matches_length,__global int *matches_offset)
{
    int idX = get_global_id(0);//blockIdx.x*blockDim.x+threadIdx.x;

    int i = idX;
    int beginSearch = idX - WINDOW_SIZE;
    if (i >= bufferSize)
    {
        return;
    }

    int length = 0;
    int offset = 0;
    int windowHead = ( idX) % WINDOW_SIZE;

    int currentOffset = 0;

    //Uncoded Lookahead optimization
    char current[MAX_CODED];
    //for (int j = 0; j < MAX_CODED && i + j < bufferSizeAdjusted; j++)
    for (int j = 0; j < MAX_CODED; j++)
    {
        current[j] = buffer[i + j];
    }

    //First WINDOW_SIZE bits will always be ' ', optimize begging where data really is
    if(beginSearch < -MAX_CODED){
        currentOffset = (beginSearch * -1) - MAX_CODED;
    }
//    char* current = buffer + i;
    int j = 0;
    while (1)
    {
        if (current[0] == (beginSearch + Wrap((currentOffset), WINDOW_SIZE) < 0? ' ': buffer[beginSearch + Wrap((currentOffset), WINDOW_SIZE)]))
        {
            /* we matched one. how many more match? */
            j = 1;

            while (
              current[j] == (beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < 0?' ':buffer[beginSearch + Wrap((currentOffset + j), WINDOW_SIZE)]) &&  
                beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < bufferSize && i + j < bufferSize)
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


__kernel void FindMatchBatchInMemoryKernel(__global unsigned char *input, int sizeInput, __global  int * breakPositions, int breakSize, __global int *matches_length, __global  int *matches_offset){
    int idX = get_global_id(0);
    if(idX >= sizeInput){
        return;
    }
    int i = idX;
    // if(idX == sizeInput -1){
    //     matches_length[i] = 0;
    //     matches_offset[i] = 0;
    // }
		
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
    

    char uncodedLookahead[MAX_CODED];
    //for (int j = 0; j < MAX_CODED && i + j < bufferSizeAdjusted; j++)
    for (int j = 0; j < MAX_CODED; j++)
    {
        uncodedLookahead[j] = input[i + j];
    }

    int thisBatchI = i - startPos ;
    int startWindowSize =max(thisBatchI - WINDOW_SIZE,0);
    // printf("i:%i lengthPos: %i, startPos:%i lastPos:%i \n",i,lengthPos,startPos,lastPos);

    // printf("i:%i startWindowSize:%i thisBatchI:%i \n",i,startWindowSize,thisBatchI);
    int longest_length = 0;
    int longest_offset = 0;
    int windowHead = thisBatchI % WINDOW_SIZE;

    for(int current = startWindowSize; current < thisBatchI; current++ ){
        if(current+ startPos < sizeInput && input[current+ startPos] == uncodedLookahead[0]){
            int j = 1;
    
            while(lastPos > current+ startPos + j
                && current + j< thisBatchI 
                && current+ startPos + j < sizeInput
                && i + j < sizeInput
                //limits the uncoded lookahead
                    && i + j < lastPos
                //find until start of uncodedLookahead
                && input[current+ startPos + j]  == uncodedLookahead[j]
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
    matches_offset[idX] = longest_offset;
    matches_length[idX] = longest_length;
}

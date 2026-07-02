import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END) # set pointer to the end of the file
    file_size = file.tell() # give the current position of the file pointer
    file.seek(0) # set pointer to the start of the file.

    chunk_size = file_size // desired_num_chunks  # estimate where is the size of byte for each chunks is about to be.

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] # get roughly idx for each boundary.
    chunk_boundaries[-1] = file_size # make sure the last position is the end of the file.

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1): # 0 and the last boundary do not need to changed, since it have to be 0 and FILE_SIZE.
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk -> form current pointer idx to/shift param we passed in.

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size # stop since is already end of file.
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token) # return the first idx found the special token. if not exist return -1
            if found_at != -1: 
                chunk_boundaries[bi] = initial_position + found_at # push the boundry to where split by special token
                break
            initial_position += mini_chunk_size # other wise repeat until we find one

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries)) # in case we not mass up order


## Usage
with open(..., "rb") as f: # rb stand for read binary | ... should replace with the actually file path name
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    # syntax: -> boundaries[:-1, boundaries[1:]] | if [0, 100, 200, 300] --> zip([0, 100, 200], [100, 200, 300]) -> [(0, 100), (100, 200), (200, 300)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore") # read each chunk accordingly
        # Run pre-tokenization on your chunk and store the counts for each pre-token


try:
    from os import (
        posix_fadvise,
        POSIX_FADV_SEQUENTIAL,
        POSIX_FADV_WILLNEED,
    )
    def os_access_hint(file_obj):
        flags = POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED
        posix_fadvise(file_obj.fileno(), 0, 0, flags)
except ImportError:
    def os_access_hint(file_obj):
        pass

def count_lines(filename):
    # reading file by chunks of the `buffer_size`
    def _buffer_gen(read_f):
        buffer_size = 128 * 1024
        buf = read_f(buffer_size)
        while buf:
            yield buf
            buf = read_f(buffer_size)
    with open(filename, 'rb') as f:
        os_access_hint(f)
        return sum(buf.count(b'\n') for buf in _buffer_gen(f.raw.read))
    return 0

"""Give every application-replay process its own immutable profiling receipt."""

import os
import sys

from profile_ncu import main

if __name__ == "__main__":
    index = sys.argv.index("--output") + 1
    sys.argv[index] = sys.argv[index].format(pid=os.getpid())
    main()

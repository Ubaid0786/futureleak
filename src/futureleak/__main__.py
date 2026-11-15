"""Allow ``python -m futureleak`` as an alternative to the ``futureleak`` command."""

import sys

from .cli import main

sys.exit(main())

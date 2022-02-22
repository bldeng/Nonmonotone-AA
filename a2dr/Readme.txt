This is the source code for the nonnegative least squares problem. The code is implemented based on a2dr (https://github.com/cvxgrp/a2dr). Please check the a2dr page for the dependencies.

Our implementation of LM-AA can be found in ./examples/paper_examples/lm_aa.py.
For better comparison, we have modified part of the implementation of the A2DR method, including:
1: We have added some codes in the "a2dr" function in ./a2dr./solver.py to record the fixed-point iteration residuals and the computational time.
2: We have modified the function "compare_total_all" in ./a2dr/tests/base_test.py to generate more figures.

To keep the style of figures in this paper consistent, we stored the generated data in matlab form and then used matlab to generate the figures in the paper. These codes can be found in ./examples/paper_examples/nnls.py line 85-89.

You can run ./examples/paper_examples/nnls.py to get these data in ./examples/paper_examples/. If you uncomment line 95 in ./examples/paper_examples/nnls.py, then the figures will pop up directly.
# mmi713_project

This repository includes the MMI713 Applied Parallel Programming on GPU Course term project files of Gazi Eren Tuncay, regarding an implementation of local maxima finding algorithm on CPU and GPU.

In all the files, the parameters and the input is set at the beginning of the file. Input data can be arranged by changing the "file_name" variable to either the folder "data" or "IRSTD-1k", the options are added as comment. 
"local_basic.py" file is the simplest version of GPU implementations and the minimum distance options are added as comment for ease of change.
"local_num.py" file is added for arranging the "num_peaks" parameter to determine the number of peaks to detect on the input.
"local_detection.py" file is the file which include the most complex version of the detection mechanism for sequential frames with the object decision module.
"local_seq.py" file is the algorithm for sequential frames but without object decision module, to be used for benchmarking.

The CPU implementations are added with the "main_" file name and the rest of the file name is the same as the GPU counterpart for corresponding functions and comparisons.

The execution can be simply done by using the following commmand, with "file_name" representing one the Python file names in the repository:

python file_name.py

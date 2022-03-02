###Run "size_constraint_k_means.py" file
Require 3 parameters in order: 

    1. Dataset file
    2. Max size of each cluster
With command line:
    
    python3 size_constraint_k_means.py para1 para2 
Example:
    
    python3 size_constraint_k_means.py Sample/sample1.txt 121
    
###Run "modified_complete_linkage.py" file
Require 3 parameters in order: 

    1. Dataset file
    2. Max diameter of cluster
    3. Max size of cluster
With command line:
    
    python3 modified_complete_linkage.py para1 para2 para3
Example:

    python3 modified_complete_linkage.py Sample/sample1.txt 111 121
    
###Notices:

    display_func in the utility.display.py file can only display datasets in up to 4 clusters.
#! /bin/bash
dirData=$1 # this is the directory where the structural and diffusion MRI data are (please note that all data should be in BIDS format)
dwis=$2 # this is a list with the conventional DWI names that the DWI data were acquired with (e.g. "DTI DKI")
brainMask=$3 # this is the path to a brain mask file provided by the user (a brain mask that was based on the T1w image of the subject)
ccases=$4 # this is a list of the types of DTI or DKI processing we are interested in
# The options you have for ccase variable are the following
# Option 1 - "LPCA"
# Option 2 - "Gibbs"
# Option 3 - "Eddy"
# Option 4 - "LPCA_Gibbs"
# Option 5 - "LPCA_Eddy"
# Option 6 - "Gibbs_Eddy"
# Option 7 - "LPCA_Gibbs_Eddy"
acqp=$5 # a text file with some acquisition information
# normally a *.txt file that is suitable for the above acqp variable would have the next two lines (of course without the symbol # in the beginning)
#0 -1 0 0.1
#0 1 0 0.1
dwi_reversed=$6 # (optional) this is the name of the reversed-enconding DWI file to-be-used as an aid to eddy-and-motion-correction method (e.g. "DTI_PA")
dwi_mask=$7 # (optional) if you have a brain mask derived by the DTI or DKI, you need to provide the whole path to the file here 
txt=$8 # this is optional - it is a text file lining up the directories where the */anat and */dwi folders are
for dwi in ${dwis}
do
for ccase in ${ccases}
do
echo "Now will work on $dwi - $ccase"
    bash $PWD/dwi_pipe/launch_dwi_pipe.bash $dirData $dwi $brainMask $ccase ${acqp} ${dwi_reversed} ${dwi_mask} $txt
done
done

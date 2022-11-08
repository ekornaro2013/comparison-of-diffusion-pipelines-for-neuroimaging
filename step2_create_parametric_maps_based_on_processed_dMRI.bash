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
# for example: ccases=["LPCA" "LPCA_Eddy"]
smooths=$5 # this is a list of the amount of Gaussian (spatial) smoothing to be applied on the processed dMRI data towards improving SNR and prior to the extraction of the diffusion paramerers
# for example: smooth=[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
dwi_mask=$6 # (optional) if you have a brain mask derived by the DTI or DKI, you need to provide the whole path to the file here 
txt=$7 # this is optional - it is a text file lining up the directories where the */anat and */dwi folders are
for dwi in ${dwis}
do
for ccase in ${ccases}
do
for smooth in ${smooths}
do
    echo "Now will work on $dwi - $ccase - $smooth"
    bash $PWD/dwi_pipe/launch_dwi_pipe_maps.bash $dirData $txt ${t1} $dwi $dwiMask $ccase $smooth
done
done
done

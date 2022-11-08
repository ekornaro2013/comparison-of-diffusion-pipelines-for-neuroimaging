## Neuroimaging scripts for processing diffusion MRI data - comparison of different dMRI pipelines
>[Evgenios N. Kornaropoulos: evgenios.kornaropoulos@med.lu.se]

This is the workflow that we designed to compare different diffurion MRI neuroimaging pipelines that can be applied on both a single-shell and mutli-shell diffusion MRI acquisitions.

## Requirement: docker installation

The only package that is required by the user to have it installed is the docker package:
https://docs.docker.com/desktop/

Docker is a platform designed to help developers build, share, and run modern applications. We handle the tedious setup, so you can focus on the code. For a tutorial of how to use docker please read the following: 
https://docs.docker.com/get-started/

## Inputs (to be provided by the use in BIDS format):

- a T1w image
- a diffusion MRI acquisition (e.g. DTI or DKI)

* please note that you should have a data-directory in your local computer (e.g. */Data/Subjects/) and each subject should be stored in a folder within that directory that has an identifier name (e.g. */Data/Subject/Subject-01 ) and then within that subject-folder you should create one last folder which you should name "anat". The T1w and T2w images or files should be placed/stored inside that "anat" folder (e.g. */Data/Subjects/Subject-01/anat/T1w.nii.gz and */Data/Subjects/Subject-01/anat/T2w.nii.gz ). Similarly, the diffusion data should have been stored as */Data/Subjects/Subject-01/dwi/DTI.nii.gz

## The different diffusion pipelines include in different combinations the following methods: 
- Denoising through the MPPCA method (Veraart et al., Neuroimage 2016)
- Gibbs ringing artefacts removal (Kellner et al, MRM 2016)
- Eddy current removal and motion correction (Andersson et al., Neuroimage 2016)

## The process (the steps you need to follow)

- Step 1: Install docker (please see above or search online on how to install docker on your computer, based on the OS your computer functions on)

- Step 2: Build a docker-image based on the Dockerfile provided to you by the zipped file
cd /path/to/extracted/files/from/the/downloaded/zipped/file
docker build -f ./Dockerfile --tags pipe:evgenios .

- Step 3: Create a docker-container based on the docker-image provided to you by the zipped file
docker run -it pipe:evgenios bash

- Step 4: Copy all the data with the T1w and T2w images inside the created docker-container (using the container's ID)
docker cp -a */Data/Subjects/Subject-01/ sdfklsdj42342:/data

* please note that sdfklsdj42342 is a random name (used just for the above example) that should be replaced by the actual docker container's ID 
* please also note that you can find the docker container's ID by typing the following in the terminal:
docker ps -a

- Step 5: Once step 5 is finished, move on launching the main bash script to process the DTI or DKI data (the following command is a command-to-type once you are within docker container's)
For example:
cd 
cd scripts
bash main.bash /data/ "DTI DKI" "brain_mask.nii.gz" "LPCA Gibbs Eddy" /path/to/acqp.txt   

## Paper to cite
Kornaropoulos, Evgenios N., Stefan Winzeck, Theodor Rumetshofer, Anna Wikstrom, Linda Knutsson, Marta M. Correia, Pia C. Sundgren, and Markus Nilsson. "Sensitivity of Diffusion MRI to White Matter Pathology: Influence of Diffusion Protocol, Magnetic Field Strength, and Processing Pipeline in Systemic Lupus Erythematosus." Frontiers in neurology 13 (2022).

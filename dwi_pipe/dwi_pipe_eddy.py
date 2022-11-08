# IMPORT BASIC PYTHON INTERFACES
import sys, os, getopt, datetime
import numpy as np
import nibabel as nib

## IMPORT SPECIFIC NIPYPE INTERFACES
import nipype.pipeline.engine as pe
from nipype.interfaces import io, fsl, dipy, ants, mrtrix3
from nipype.algorithms import metrics
from nipype.interfaces.utility import IdentityInterface, Merge, Rename
from nipype.workflows.dmri.fsl.artifacts import remove_bias
from nipype import config

config.enable_debug_mode()

# HELPER INTERFACES
# appends path where "structural_pipeline.py" is stored, keep helper interfaces in there
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helperDiffusionPipeline as HelperInterfaces
#from PrintColours import colours
from dipy.io import read_bvals_bvecs


def _check_nb_bvals(fbvals, fbvecs):
        """ Takes bval and bvec file and computes the number of bvals,
                unique bvals and the number of unique bvals.
                fbvals: string to bval file
                fbvecs: string to bvec file
        """
        bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
        nbBvals = len(bvals)
        uniqueBvals = np.unique(np.round(bvals/100)*100)
        if uniqueBvals[0]==0: uniqueBvals = uniqueBvals[1:]
        nbUniqueBvals = len(uniqueBvals)

        return uniqueBvals, nbUniqueBvals, nbBvals


def BuildPipeLine():
        """ Building nipype processing pipeLine
        """
        #-----------------------------------------------------------------------------------------------------#
        # PARSE AND CHECK ARGUMENTS
        #-----------------------------------------------------------------------------------------------------#
        try:
                opts, args = getopt.getopt(sys.argv[1:],'-ho:c:d:b:r:e:p:s:m:a:', ['help', 'outDir=','name=','DWI=','bval=','bvec=', 'DWIMask=', 'ExtraB0=', 'T1=','T1mask=', 'acqp='])
        except getopt.GetoptError as err:
                print(err) # will print(something like "option <> not recognized"
                print('usage: diffusion_pipeline.py',
                        '\n -o <output directory>',
                        '\n -c <name_of_nipype>',
                        '\n -d <directory/to/DWI.nii>',
                        '\n -b <directory/to/DWI.bval>',
                        '\n -r <directory/to/DWI.bvec>',
                        '\n -e <directory/to/DWImask.nii>',
                        '\n -p  <directory/to/extra_b0.nii>'
                        '\n -s <directory/to/T1.nii>',
                        '\n -m <directory/to/T1mask.nii>',
                        '\n -a <directory/to/acqp.txt>')
                        #'\n -i <directory/to/index.txt>')
                sys.exit(2)

        print(opts)
        for opt, arg in opts:
                if opt in ('-h','--help') or len(opts)!=10:
                        print('usage: diffusion_pipeline.py',
                        '\n -o <output directory>',
                        '\n -c <name_of_nipype>',
                        '\n -d <directory/to/DWI.nii>',
                        '\n -b <directory/to/DWI.bval>',
                        '\n -r <directory/to/DWI.bvec>',
                        '\n -e <directory/to/DWImask.nii>',
                        '\n -p  <directory/to/extra_b0.nii>'
                        '\n -s <directory/to/T1.nii>',
                        '\n -m <directory/to/T1mask.nii>',
                        '\n -a <directory/to/acqp.txt>')
                        sys.exit(2)

                elif opt in ('-o','--outDir'):
                        expDir = arg
                elif opt in ('-c','--name'):
                        nipype_name = arg
                elif opt in ('-d','--DWI'):
                        DWIfile = arg
                elif opt in ('-b','--bval'):
                        Bvalfile = arg
                elif opt in ('-r','--bvec'):
                        Bvecfile = arg
                elif opt in ('-e','--DWImask'):
                        DWImaskFile = arg
                elif opt in ('-p','--PE_b0'):
                        ExtraB0File = arg
                elif opt in ('-s','--T1'):
                        T1file = arg
                elif opt in ('-m','--T1mask'):
                        T1maskFile = arg
                elif opt in ('-a','--acqp'):
                        acqpFile = arg
                else:
                        assert False, "unhandled option"

        # check if a DWI mask was specified
        if DWImaskFile=='None':
                print('No DWImaks was provided')
                DWImaskFile = None
                outputFolderName = 'diffusion_dwi'
        else:
                outputFolderName = 'diffusion_dwi_manualMask'

        # check if an extra b0 file was specified
        if ExtraB0File=='None':
                print('No extra b0 file was provided')
                ExtraB0File = None


        #-----------------------------------------------------------------------------------------------------#
        # INPUT SOURCE NODES
        #-----------------------------------------------------------------------------------------------------#
        #print(colours.green + "Create Source Nodes." + colours.ENDC
        infoSource = pe.Node(IdentityInterface(fields=['dwi_file', 'bval', 'bvec', 'dwi_mask', 'extra_b0', 'T1_file', 'mask', 'acqp']), name='infosource')
        infoSource.inputs.dwi_file = DWIfile
        infoSource.inputs.bval = Bvalfile
        infoSource.inputs.bvec = Bvecfile
        infoSource.inputs.dwi_mask = DWImaskFile
        infoSource.inputs.extra_b0 = ExtraB0File
        infoSource.inputs.T1_file = T1file
        infoSource.inputs.mask = T1maskFile
        infoSource.inputs.acqp = acqpFile

        #-----------------------------------------------------------------------------------------------------#
        # TERMINAL OUTPUT
        #-----------------------------------------------------------------------------------------------------#
        # ASSESS BVAL/BVEC CHARACTERISATION
        uniqueBvals, nbUniqueBvals, nbBvals = _check_nb_bvals(Bvalfile, Bvecfile)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Number of Bvals: ', nbBvals)
        print('Number of Unique Bvals: ', nbUniqueBvals)
        print('Unique Bvals: ', uniqueBvals)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        # CREATE OUTPUT TEXT FILE that lists the input files for later reference
        # if file already exists, file is deleted and recreated
        txtFile = os.path.join(expDir,'inputFiles_' + outputFolderName + '_pip.txt')
        if os.path.isfile(txtFile):
                print('Old <inputFiles_' + outputFolderName + '_pip.txt> removed.')
                os.remove(txtFile)

        # check if outputdolder exists, otherwise create
        if not os.path.exists(os.path.join(expDir, '%s_pipe_eddy' % nipype_name, outputFolderName)):
                os.makedirs(os.path.join(expDir, '%s_pipe_eddy' % nipype_name, outputFolderName))

        # check if index file is avialble, if not create a new one
        eddyIndexTxtFile = os.path.join(expDir, '%s_pipe_eddy' % nipype_name, outputFolderName, 'eddy_index.txt')
        if os.path.isfile(eddyIndexTxtFile):
                print('Use index file: ', eddyIndexTxtFile)
        else:
                eddyIndex = np.ones((1, nbBvals), int)
                np.savetxt(eddyIndexTxtFile, eddyIndex, fmt='%.i')
                print('Save index file: ', eddyIndexTxtFile)

        infoSource.inputs.index = eddyIndexTxtFile

        # PRINT INPUT INFORTMATION TO SCREEN AND SAVE IN OUTPUT FILE
        pipelineInfo = '-------------------' + \
                '\nThis pipeline version v0.2 ran on ' + str(datetime.date.today()) + \
                '\nPlease include in co-author list:' + \
                '\nEvgenios Kornaropoulos, Division of Anaesthesia, Department of Medicine, University of Cambridge, Cambridge, UK' + \
                '\nStefan Winzeck, Division of Anaesthesia, Department of Medicine, University of Cambridge, Cambridge, UK' + \
                '\nMarta M. Correia, MRC Cognition and Brain Sciences Unit, University of Cambridge, Cambridge, UK' + \
                '\n-------------------'
        print(pipelineInfo)
        #print(colours.green + 'Input Files' + colours.ENDC)

        f=open(txtFile,'w')
        f.write(pipelineInfo + '\n')
        #f.write(colours.green + 'Input Files' + colours.ENDC + '\n')
#       for tag, inputFile in zip(['DWI', 'bvals', 'bvecs', 'DWI Mask', 'Extra b0', 'T1w', 'T1w Mask', 'ACQP', 'INDEX'],
#                       [DWIfile, Bvalfile, Bvecfile, DWImaskFile, ExtraB0File, T1file, T1maskFile, acqpFile, eddyIndexTxtFile]):
#               if inputFile==None:
#                       print('{0:8} | '.format(tag), colours.red + str(inputFile) + colours.ENDC)
#                       output = '{0:8} | '.format(tag) + colours.red + str(inputFile) + colours.ENDC  + '\n'
#
#               else:
#                       print('{0:8} | '.format(tag), inputFile)
#                       output = '{0:8} | '.format(tag) + str(inputFile) + '\n'
#
#               f.write(output)

        print('{0:8} | '.format('outDir'), expDir)
        print('-------------------')

        output = '{0:8} | '.format('outDir') + expDir + '\n-------------------\n'
        f.write(output)
        f.close()

        # ASSESS 3rd DIMENSION OF DWI IMAGE
        # topup runs with given paprameters only when dimensions are even
        # if dimesnions are odd, cut off first slice, only do so when extra b0 is provided
        # otherwise topup is not applied anyways, so no cropping will take place
        valueForCutOfAxialSlice_X = 0
        valueForCutOfAxialSlice_Y = 0
        valueForCutOfAxialSlice_Z = 0

        if ExtraB0File!=None:
                print(ExtraB0File)
                extraB0ImageDimX, extraB0ImageDimY, extraB0ImageDimZ = nib.load(ExtraB0File).get_fdata().shape
                if extraB0ImageDimX % 2!=0:
                        valueForCutOfAxialSlice_X = 1
                        print('DWI volume has odd x-dimension: Cut of lowest slice')
                if extraB0ImageDimY % 2!=0:
                        valueForCutOfAxialSlice_Y = 1
                        print('DWI volume has odd y-dimension: Cut of lowest slice')
                if extraB0ImageDimZ % 2!=0: #
                        valueForCutOfAxialSlice_Z = 1
                        print('DWI volume has odd z-dimension: Cut of lowest slice')

        #-----------------------------------------------------------------------------------------------------#
        # WORKFLOW
        #-----------------------------------------------------------------------------------------------------#
        #print(colours.green + "Create Workflow" + colours.ENDC)
        preproc = pe.Workflow(name=outputFolderName)
        preproc.base_dir = os.path.join(expDir, '%s_pipe_eddy' % nipype_name)
        preproc.config['execution'] = {'remove_unnecessary_outputs' : 'true', 'keep_inputs' : 'true', 'stop_on_first_crash' : 'true'}
#       preproc.config['execution'] = {'remove_unnecessary_outputs' : 'false', 'keep_inputs' : True, 'remove_unnecessary_outputs' : True, 'stop_on_first_crash' : True}

        #-----------------------------------------------------------------------------------------------------#
        # PROCESSING NODES
        #print(colours.green + 'Create Processing Nodes.' + colours.ENDC)
        #-----------------------------------------------------------------------------------------------------#
        # MASK T1 FILE
        maskT1 = pe.Node(fsl.ApplyMask(), name='mask_T1')
        maskT1.inputs.output_type = 'NIFTI_GZ'

        # SELECT 1st VOLUME
        selectLowB = pe.Node(fsl.ExtractROI(), name='select_b0')
        selectLowB.inputs.t_min = 0
        selectLowB.inputs.t_size = 1
        selectLowB.inputs.roi_file = 'b0.nii.gz'
        selectLowB.inputs.output_type = 'NIFTI_GZ'

        # STRIP AWAY THE B0 VOLUME AFTER ARTIFACT CORRECTION
        selectCorrectedDWI = pe.Node(fsl.ExtractROI(), name='select_corrected_DWI')
        selectCorrectedDWI.inputs.t_min = 0
        selectCorrectedDWI.inputs.t_size = nbBvals
        selectCorrectedDWI.inputs.roi_file = 'corrected_DWI.nii.gz'
        selectCorrectedDWI.inputs.output_type = 'NIFTI_GZ'

        # INTERFACES TO PREPARE FOR AND USE TOPUP
        if ExtraB0File!=None:
                # if odd dimesions cut of slice=0 from all three planes of DWI image
                cutDWI = pe.Node(fsl.ExtractROI(), name='cut_DWI')
                cutDWI.inputs.x_min = valueForCutOfAxialSlice_X
                cutDWI.inputs.x_size = -1
                cutDWI.inputs.y_min = valueForCutOfAxialSlice_Y
                cutDWI.inputs.y_size = -1
                cutDWI.inputs.z_min = valueForCutOfAxialSlice_Z
                cutDWI.inputs.z_size = -1
                cutDWI.inputs.roi_file = 'cutDWI.nii.gz'
                cutDWI.inputs.output_type = 'NIFTI_GZ'

                # if odd dimesions cut of slice=0 from all three planes of extra b0 image
                cutB0 = pe.Node(fsl.ExtractROI(), name='cut_extraB0')
                cutB0.inputs.x_min = valueForCutOfAxialSlice_X
                cutB0.inputs.x_size = -1
                cutB0.inputs.y_min = valueForCutOfAxialSlice_Y
                cutB0.inputs.y_size = -1
                cutB0.inputs.z_min = valueForCutOfAxialSlice_Z
                cutB0.inputs.z_size = -1
                cutB0.inputs.roi_file = 'cutExtraB0.nii.gz'
                cutB0.inputs.output_type = 'NIFTI_GZ'

                #make list of DWI and extra b0
                listDWI = pe.Node(Merge(2), name='list_DWI')

                #make list of b0 and rebb0
                listB0 = pe.Node(Merge(2), name='list_b0')

                # merge DWI and extra b0 volume for artifact correction (i.e. LPCA & Gibbs)
                mergeOppostitePE = pe.Node(fsl.Merge(), name='merge_opposite_PE')
                mergeOppostitePE.inputs.dimension = 't'
                mergeOppostitePE.inputs.merged_file = 'merged_DWI_extrab0.nii.gz'
                mergeOppostitePE.inputs.output_type = 'NIFTI_GZ'

                # select extra b0 volume after artifact correction
                selectExtraB = pe.Node(fsl.ExtractROI(), name='select_revB0')
                selectExtraB.inputs.t_min = nbBvals
                selectExtraB.inputs.t_size = 1
                selectExtraB.inputs.roi_file = 'revB0.nii.gz'
                selectExtraB.inputs.output_type = 'NIFTI_GZ'

                # merge both corrected b0s with oposite phase encoding directions
                mergeOppostiteB0 = pe.Node(fsl.Merge(), name='merge_opposite_b0')
                mergeOppostiteB0.inputs.dimension = 't'
                mergeOppostiteB0.inputs.merged_file = 'merged_b0_revB0.nii.gz'
                mergeOppostiteB0.inputs.output_type = 'NIFTI_GZ'

                # apply topup
                topup = pe.Node(fsl.TOPUP(), name='topup')
                topup.inputs.output_type = "NIFTI_GZ"


        # HELPER NODE TO SELECT DWI or DWI_B0 file
        renameDWI =  pe.Node(Rename(), name='select_dwi')
        renameDWI.inputs.format_string = 'selectedDWI.nii.gz'

        # DENOISING
        lpca = pe.Node(mrtrix3.DWIDenoise(), name='LPCA')

        # GIBBS RINGING REMOVAL
        gibbs = pe.Node(HelperInterfaces.MRTRIX3GibbsRemoval(), name='gibbs_removal')

        # IF NOT SPECIFIED GET b0 MASK
        if DWImaskFile==None:
                selectCorrectedB = pe.Node(fsl.ExtractROI(), name='select_corrected_b0')
                selectCorrectedB.inputs.t_min = 0
                selectCorrectedB.inputs.t_size = 1
                selectCorrectedB.inputs.roi_file = 'corrected_b0.nii.gz'
                selectCorrectedB.inputs.output_type = 'NIFTI_GZ'

                ants_reg_LOWBtoT1 = pe.Node(ants.Registration(), name='ants_reg_LOWBtoT1')
                ants_reg_LOWBtoT1.inputs.dimension = 3
                ants_reg_LOWBtoT1.inputs.transforms = ['Rigid']
                ants_reg_LOWBtoT1.inputs.transform_parameters = [(0.25,)]
                ants_reg_LOWBtoT1.inputs.metric = ['MI']
                ants_reg_LOWBtoT1.inputs.initial_moving_transform_com = 1
                ants_reg_LOWBtoT1.inputs.metric_weight = [1]
                ants_reg_LOWBtoT1.inputs.smoothing_sigmas = [[4, 2, 1, 0]]
                ants_reg_LOWBtoT1.inputs.shrink_factors = [[8, 4, 2, 1]]
                ants_reg_LOWBtoT1.inputs.sigma_units = ['mm']
                ants_reg_LOWBtoT1.inputs.radius_or_number_of_bins = [32]
                ants_reg_LOWBtoT1.inputs.number_of_iterations = [[1000, 500, 250, 100]]
                ants_reg_LOWBtoT1.inputs.convergence_threshold = [1.e-8]
                ants_reg_LOWBtoT1.inputs.convergence_window_size = [10]
                ants_reg_LOWBtoT1.inputs.sampling_strategy = ['Regular']
                ants_reg_LOWBtoT1.inputs.sampling_percentage = [0.25]
                ants_reg_LOWBtoT1.inputs.use_histogram_matching = [False]
                ants_reg_LOWBtoT1.inputs.write_composite_transform = True
                ants_reg_LOWBtoT1.inputs.collapse_output_transforms = True
                ants_reg_LOWBtoT1.inputs.initialize_transforms_per_stage = False
                ants_reg_LOWBtoT1.inputs.output_transform_prefix = 'ants_rig_'
                ants_reg_LOWBtoT1.inputs.output_warped_image = 'ants_rig_LOWB.nii.gz'

                ants_warpT1mask = pe.Node(ants.ApplyTransforms(), name='ants_warpMask_T1toDWI')
                ants_warpT1mask.inputs.dimension = 3
                ants_warpT1mask.inputs.output_image = 'T1_mask_dwispace.nii.gz'
                ants_warpT1mask.inputs.interpolation = 'NearestNeighbor'

                maskName = 'ANTS_T1_brain_mask.nii.gz'
        else:
                maskName = 'manual_brain_mask.nii.gz'

        # HELPER NODE TO SELECT COMPUTED OR MANUAL MASK
        rename = pe.Node(Rename(), name='select_dwi_mask')
        rename.inputs.format_string = maskName

        # HEAD MOTION & EDDY CURRENT CORRECTION
        eddy = pe.Node(fsl.Eddy(), name='eddy')
        eddy.inputs.interp = 'spline'
        eddy.inputs.use_cuda = True
        eddy.inputs.is_shelled = True
        eddy.inputs.num_threads = 4
        eddy.inputs.args = '--data_is_shelled --ol_nstd=5'
        eddy.inputs.output_type = 'NIFTI_GZ'

        # REMOVE BIAS
        convertMask = pe.Node(mrtrix3.MRConvert(), name='convert_brain_mask')
        convertMask.inputs.out_file = 'DWI_brain_mask_comptessed.nii.gz'
        convertMask.inputs.args = '-datatype int16'
        bias = remove_bias()

        # ALTERNATIVE BRAIN MASKING (for comparsision to ANTS or manual brain main mask)
        mrtrixBrainMask = pe.Node(HelperInterfaces.MRTRIX3BrainMask(), name='mrtrix_brain_mask')

        
        # WM PARCELLATION WITH TractSeg, version 3
        # check the gradient directions
        mrtrixGradCheck = pe.Node(HelperInterfaces.MRTRIX3GradCheck(), name='mrtrix_grad_check')
        # raw tract segmentation after mrtrix 
        tractSeg = pe.Node(HelperInterfaces.RawTractSeg(), name='tract_seg')
        # calculate metrics of the tracts 
        tractMetrics = pe.Node(HelperInterfaces.TractMetrics(), name='tract_metrics')
        # QC - mosaic of tracts 
        mosaicTracts = pe.Node(HelperInterfaces.MosaicTracts(), name='QC_tracts_mosaic')

        # ADDITIONAL DWI MAPS
        multiplyer = pe.MapNode(fsl.ImageMaths(), iterfield=['op_string', 'out_file'], name='b_multiplier', synchronize=True)
        multiplyer.inputs.op_string = ['-mul ' + str(-1*b) for b in uniqueBvals]
        multiplyer.inputs.out_file = ['multipliedMD_b' + str(int(b)).zfill(4) + '.nii.gz' for b in uniqueBvals]
        multiplyer.inputs.output_type = 'NIFTI_GZ'

        traceMap = pe.MapNode(fsl.ImageMaths(), iterfield=['in_file', 'out_file'], name='trace_map', synchronize=True)
        traceMap.inputs.op_string = '-exp -mul'
        traceMap.inputs.out_file =  ['traceMap_b' + str(int(b)).zfill(4)  + '.nii.gz' for b in uniqueBvals]
        traceMap.inputs.output_type = 'NIFTI_GZ'

        estmRSH = pe.Node(dipy.EstimateResponseSH(), name='estimate_response_SH')
        tensorMode = pe.Node(dipy.TensorMode(), name='tensor_mode')
        powerMap = pe.Node(dipy.APMQball(), name='power_Map')
        renamePowerMap = pe.Node(Rename(), name='rename_power_map')
        renamePowerMap.inputs.format_string = 'powermap.nii.gz'

        # GET DWI MASKS
        # dwimask is a mask that excludes L1<0 and FA>1
        dwiMask = pe.Node(HelperInterfaces.DWIMask(), name='dwifit_mask')

        # MULTI MODAL REGISTRATION: FA & PowerMap -> T1w & T1w
        mergeFAPM = pe.Node(Merge(2), name='merge_FAPM')
        mergeT1T1 = pe.Node(Merge(2), name='merge_T1T1')

        ants_rig_DWItoT1 = pe.Node(ants.Registration(), name='ants_rig_DWItoT1')
        ants_rig_DWItoT1.inputs.dimension = 3
        ants_rig_DWItoT1.inputs.transforms = ['Rigid']
        ants_rig_DWItoT1.inputs.initial_moving_transform_com = 1
        ants_rig_DWItoT1.inputs.metric = [['MI', 'MI']]
        ants_rig_DWItoT1.inputs.metric_weight = [[.5, .5]]
        ants_rig_DWItoT1.inputs.sampling_strategy = [['Regular', 'Regular']]
        ants_rig_DWItoT1.inputs.sampling_percentage = [[.25, .25]]
        ants_rig_DWItoT1.inputs.radius_or_number_of_bins = [[32, 32]]
        ants_rig_DWItoT1.inputs.transform_parameters = [(.25,)]
        ants_rig_DWItoT1.inputs.number_of_iterations = [[1000, 500, 250, 100]]
        ants_rig_DWItoT1.inputs.shrink_factors = [[8, 4, 2, 1]]
        ants_rig_DWItoT1.inputs.smoothing_sigmas = [[4, 2, 1, 0]]
        ants_rig_DWItoT1.inputs.sigma_units = ['mm']
        ants_rig_DWItoT1.inputs.convergence_threshold = [1.e-8]
        ants_rig_DWItoT1.inputs.convergence_window_size = [10]
        ants_rig_DWItoT1.inputs.use_histogram_matching = [False]
        ants_rig_DWItoT1.inputs.write_composite_transform = True
        ants_rig_DWItoT1.inputs.collapse_output_transforms = True
        ants_rig_DWItoT1.inputs.output_transform_prefix = "ants_rig_DWItoT1_"
        ants_rig_DWItoT1.inputs.output_warped_image = 'FA_DWI_anatSpace.nii.gz'
        ants_rig_DWItoT1.inputs.interpolation = 'WelchWindowedSinc'

        mergeDiffMaps = pe.Node(Merge(4), name='merge_DiffMaps')

        # WARP DWI MAPS TO T1 SPACE
        ants_warpDiffMaps = pe.MapNode(ants.ApplyTransforms(), name='ants_warpDiffMaps',iterfield=['input_image', 'output_image'])
        ants_warpDiffMaps.inputs.dimension = 3
        ants_warpDiffMaps.inputs.output_image = ['MD_DWI_anatSPace.nii.gz', 'L1_DWI_anatSPace.nii.gz', 'L2_DWI_anatSPace.nii.gz','L3_DWI_anatSPace.nii.gz']
        ants_warpDiffMaps.inputs.interpolation = 'WelchWindowedSinc'

        ants_warpPM = pe.Node(ants.ApplyTransforms(), name='ants_warpPM')
        ants_warpPM.inputs.dimension = 3
        ants_warpPM.inputs.output_image = 'APM_DWI_anatSpace.nii.gz'
        ants_warpPM.inputs.interpolation = 'WelchWindowedSinc'

        ants_warpS0 = pe.Node(ants.ApplyTransforms(), name='ants_warpS0')
        ants_warpS0.inputs.dimension = 3
        ants_warpS0.inputs.output_image = 'S0_DWI_anatSpace.nii.gz'
        ants_warpS0.inputs.interpolation = 'WelchWindowedSinc'

        ants_warpTraceMaps = pe.MapNode(ants.ApplyTransforms(), iterfield=['input_image', 'output_image'], name='ants_warpTraceMaps', synchronize=True)
        ants_warpTraceMaps.inputs.dimension = 3
        ants_warpTraceMaps.inputs.output_image = ['traceMap_b' + str(int(b)).zfill(4)  + '_DWI_anatSpace.nii.gz' for b in uniqueBvals]
        ants_warpTraceMaps.inputs.interpolation = 'WelchWindowedSinc'


        # RENAME OUTPUT FILES
        renameEddyDWI = pe.Node(Rename(), name='rename_eddy_dwi')
        renameEddyDWI.inputs.format_string = 'DWI_denoised_gibbs_eddy.nii.gz'

        renameEddyBvec = pe.Node(Rename(), name='rename_eddy_bvec')
        renameEddyBvec.inputs.format_string = 'DWI_corrected.bvec'
        renameBval = pe.Node(Rename(), name='rename_bval')
        renameBval.inputs.format_string = 'DWI_corrected.bval'

        renameBias = pe.Node(Rename(), name='rename_bias')
        renameBias.inputs.format_string = 'DWI_corrected.nii.gz'

        # GET ORIGINAL FILES SIZE original file
        selectDenoised = pe.Node(fsl.ExtractROI(), name='select_denoised')
        selectDenoised.inputs.t_min = 0
        selectDenoised.inputs.t_size = nbBvals
        selectDenoised.inputs.roi_file = 'DWI_denoised.nii.gz'
        selectDenoised.inputs.output_type = 'NIFTI_GZ'

        selectGibbs = pe.Node(fsl.ExtractROI(), name='select_gibbs')
        selectGibbs.inputs.t_min = 0
        selectGibbs.inputs.t_size = nbBvals
        selectGibbs.inputs.roi_file = 'DWI_denoised_gibbs.nii.gz'
        selectGibbs.inputs.output_type = 'NIFTI_GZ'

        # DATASINK
        dataSink = pe.Node(io.DataSink(parameterization=False), name='datasink')
        dataSink.inputs.base_directory = os.path.join(expDir, '%s_pipe_eddy' % nipype_name)
        dataSink.inputs.container = 'DATASINK'

        # MULTI-SHELL
        fwDWI = pe.Node(HelperInterfaces.FreeWaterElimination(), name='free_water_DWI')
        dwiThr = pe.Node(fsl.ApplyMask(), name='dwifit_removeFW')
        #ivim = pe.Node(HelperInterfaces.IntraVoxelIncoherentMotion(), name='inravoxel_incoherent_motion')  -> potentially useful for low bvals <1000

        #-----------------------------------------------------------------------------------------------------#
        # QUALITY COTROL NODES
        #-----------------------------------------------------------------------------------------------------#
        #print(colours.green + 'Create QC Nodes.' + colours.ENDC)

        # MOSAIC IMAGE FOR BRAIN MASK QC
        ants_rgb_mask = pe.Node(ants.ConvertScalarImageToRGB(), name='ants_rgb_mask')
        ants_rgb_mask.inputs.dimension = 3
        ants_rgb_mask.inputs.colormap = 'red'
        ants_rgb_mask.inputs.minimum_input = 0
        ants_rgb_mask.inputs.maximum_input = 1

        mosaic_masks = pe.Node(ants.CreateTiledMosaic(), name='QC_mask_mosaic')
        mosaic_masks.inputs.direction = 2
        mosaic_masks.inputs.alpha_value = 0.2
        mosaic_masks.inputs.flip_slice = '0x1'
        mosaic_masks.inputs.slices = '[3, mask, mask]'
        mosaic_masks.inputs.output_image = 'DWI_mask.png'
        mosaic_masks.inputs.tile_geometry = '3x10'

        # QUALITATIVE FA COREGISTRATION QC: Mosaic image of FA overlay on T1
        ants_rgb_fa = pe.Node(ants.ConvertScalarImageToRGB(), name='ants_rgb_fa')
        ants_rgb_fa.inputs.dimension = 3
        ants_rgb_fa.inputs.colormap = 'jet'
        ants_rgb_fa.inputs.minimum_input = 0
        ants_rgb_fa.inputs.maximum_input = 1

        mosaic_fa = pe.Node(ants.CreateTiledMosaic(), name='QC_fa_mosaic')
        mosaic_fa.inputs.direction = 2
        mosaic_fa.inputs.alpha_value = 0.2
        mosaic_fa.inputs.flip_slice = '0x1'
        mosaic_fa.inputs.slices = '[3, mask+20, mask-20]'
        mosaic_fa.inputs.output_image = 'FA_DWI_coreg.png'

        # QUNANTITATIVE FA COREGISTRATION QC: Normalised cross correlation
        coreg_similarity = pe.Node(HelperInterfaces.ComputeNCC(), name='coreg_similarity')

        # HEAD MOTION QC
        eddyQC = pe.Node(HelperInterfaces.EddyQC(), name='QC_eddy')

        # SINGAL-TO-NOISE-RATION QC
        diff_noise = pe.Node(fsl.ImageMaths(), name='diff_noise')
        diff_noise.inputs.op_string = "-sub"
        diff_noise.inputs.out_file = "LPCA_difference_noise.nii.gz"
        diff_noise.inputs.output_type = "NIFTI_GZ"
        compute_snr = pe.Node(HelperInterfaces.ComputeSNR(), name='SNR')

        # BRAIN MASK COMPARISON
        QC_brainMasks = pe.Node(HelperInterfaces.QCcompareBrainMasks(), name='QC_brainMasks')

        # QC REPORT TO STORE ALL THE INFORMATION
        QC_report = pe.Node(HelperInterfaces.QCReport(), name='QC_report')


        #-----------------------------------------------------------------------------------------------------#
        # CONNECT INPUT AND OUTPUT NODES
        #-----------------------------------------------------------------------------------------------------#
        # BRAIN MASK NODES: Depended whether a brain maks was provided or not, a brain mask is comupted via
        # rig resgistration of b0 to T1 images
        #print(colours.green + 'Connect Nodes.' + colours.ENDC)
        if DWImaskFile==None:
                # if DWI mask was no specified compute it
                preproc.connect([
                        (renameDWI, selectCorrectedB, [('out_file', 'in_file')]),
                        (selectCorrectedB, ants_reg_LOWBtoT1, [('roi_file', 'moving_image')]),
                        (maskT1, ants_reg_LOWBtoT1, [('out_file', 'fixed_image')]),
                        (selectCorrectedB, ants_warpT1mask, [('roi_file', 'reference_image')]),
                        (ants_reg_LOWBtoT1, ants_warpT1mask, [('inverse_composite_transform', 'transforms')]),
                        (infoSource, ants_warpT1mask, [('mask', 'input_image')]),
                        (ants_warpT1mask, rename, [('output_image', 'in_file')]),

                        (maskT1, coreg_similarity, [('out_file', 'in_file1')]),
                        (ants_reg_LOWBtoT1, coreg_similarity, [('warped_image', 'in_file2')]),
                        (rename, coreg_similarity, [('out_file', 'in_mask')]),
                ])
        else:
                # if DWI mask was specified select that one
                preproc.connect([
                        (infoSource, rename, [('dwi_mask', 'in_file')]),
                ])

        # DISTORTION CORRECTIONS NODES: Dependent whether an extra b0 was provided or not, the pipeline
        # performs distorition correction
        if ExtraB0File!=None:
                preproc.connect([
                        # possibly cut of slices in case it dimensions are odd to use topup default parameters
                        (infoSource, cutDWI, [('dwi_file', 'in_file')]),
                        (infoSource, cutB0, [('extra_b0', 'in_file')]),

                        # create list of both files to merge later
                        (cutDWI, listDWI, [('roi_file', 'in1')]),
                        (cutB0, listDWI, [('roi_file', 'in2')]),

                        # merge a diffusion data and extra b0
                        (listDWI, mergeOppostitePE, [('out', 'in_files')]),
                        (mergeOppostitePE, renameDWI, [('merged_file', 'in_file')]),

                        # after correction (noise & Gibbs) select b0 volumes only and merge to one file
                        (renameDWI, selectLowB, [('out_file', 'in_file')]),
                        (renameDWI, selectExtraB, [('out_file', 'in_file')]),
                        (selectLowB, listB0, [('roi_file', 'in1')]),
                        (selectExtraB, listB0, [('roi_file', 'in2')]),
                        (listB0, mergeOppostiteB0, [('out', 'in_files')]),

                        # apply topup
                        (mergeOppostiteB0, topup, [('merged_file', 'in_file')]),
                        (infoSource, topup, [('acqp', 'encoding_file')]),

                        # apply eddy with topup information
                        (renameDWI, selectCorrectedDWI, [('out_file', 'in_file')]),
                        (selectCorrectedDWI, eddy, [('roi_file', 'in_file')]),
                        (infoSource, eddy, [('bval', 'in_bval')]),
                        (infoSource, eddy, [('bvec', 'in_bvec')]),
                        (infoSource, eddy, [('acqp', 'in_acqp')]),
                        (infoSource, eddy, [('index', 'in_index')]),
                        (topup, eddy, [('out_fieldcoef', 'in_topup_fieldcoef')]),
                        (topup, eddy, [('out_movpar', 'in_topup_movpar')]),
                        (rename, eddy, [('out_file', 'in_mask')]),
                ])
        else:
                preproc.connect([
                        # apply eddy without topup information
                        (infoSource, renameDWI, [('dwi_file', 'in_file')]),
                        (renameDWI, eddy, [('out_file', 'in_file')]),
                        (infoSource, eddy, [('bval', 'in_bval')]),
                        (infoSource, eddy, [('bvec', 'in_bvec')]),
                        (infoSource, eddy, [('acqp', 'in_acqp')]),
                        (infoSource, eddy, [('index', 'in_index')]),
                        (rename, eddy, [('out_file', 'in_mask')]),
                ])


        preproc.connect([
                # MASK T1 FILE
                (infoSource, maskT1, [('T1_file', 'in_file')]),
                (infoSource, maskT1, [('mask', 'mask_file')]),


                # EXTRA BRAIN MASK FOR COMPARSION
                (eddy, mrtrixBrainMask, [('out_corrected', 'in_file')]),
                (infoSource, mrtrixBrainMask, [('bval', 'in_bval')]),
                (eddy, mrtrixBrainMask, [('out_rotated_bvecs', 'in_bvec')]),


                # WM PARCELLATION - TractSeg. Version 3 
                # gradcheck
                (eddy, mrtrixGradCheck, [('out_corrected', 'in_file')]),
                (infoSource, mrtrixGradCheck, [('bval', 'in_bvals')]),
                (eddy, mrtrixGradCheck, [('out_rotated_bvecs', 'in_bvecs')]),
                # TractSeg
                (eddy, tractSeg, [('out_corrected', 'in_file')]),
                (mrtrixGradCheck, tractSeg, [('out_bvals', 'in_bvals')]),
                (mrtrixGradCheck, tractSeg, [('out_bvecs', 'in_bvecs')]),
                (rename, tractSeg, [('out_file', 'in_mask')]),

                (rename, QC_brainMasks, [('out_file', 'in_file1')]),
                (mrtrixBrainMask, QC_brainMasks, [('out_mask', 'in_file2')]),

                (coreg_similarity, QC_report, [('NCC', 'in_NCC')]),
                (eddy, eddyQC, [('out_movement_rms', 'in_eddy_rms')]),
                (QC_brainMasks, QC_report, [('out_volume1', 'mask_volume1')]),
                (QC_brainMasks, QC_report, [('out_volume2', 'mask_volume2')]),
                (QC_brainMasks, QC_report, [('out_volumeRatio', 'mask_volumeRatio')]),
                (eddyQC, QC_report, [('out_avg_total_motion', 'eddy_avg_total_motion')]),
                (eddyQC, QC_report, [('out_max_total_motion', 'eddy_max_total_motion')]),
                (eddyQC, QC_report, [('out_avg_relative_motion', 'eddy_avg_relative_motion')]),
                (eddyQC, QC_report, [('out_max_relative_motion', 'eddy_max_relative_motion')]),
        # ---------------------------------------
                #  DATA SINK
                # ---------------------------------------
                # RENAME FILES
                (eddy, renameEddyDWI, [('out_corrected', 'in_file')]),
                (eddy, renameEddyBvec, [('out_rotated_bvecs', 'in_file')]),
                (infoSource, renameBval, [('bval', 'in_file')]),


                # FSL EDDY
                (renameEddyDWI, dataSink, [('out_file', 'DWIspace.dwi_proc.@eddy')]),
                (renameEddyBvec, dataSink, [('out_file', 'DWIspace.dwi_proc.@bvecs')]),
                (renameBval, dataSink, [('out_file', 'DWIspace.dwi_proc.@bvals')]),


                # MASKS
                (rename, dataSink, [('out_file', 'DWIspace.masks.@brain')]),
                (mrtrixBrainMask, dataSink, [('out_mask', 'DWIspace.masks.@mask')]),

                
                # WM PARCELLATION - Version 3
                (tractSeg, dataSink, [('out_probability_atlas', 'DWIspace.WM_parcellation.@probability')]),
                (tractSeg, dataSink, [('out_binary_atlas', 'DWIspace.WM_parcellation.@segmentation')]),
                (tractSeg, dataSink, [('out_peaks', 'DWIspace.WM_parcellation.@peaks')]), 

                # QCs
                (QC_report, dataSink, [('out_csv', 'DWIspace.QC.@report')]),

        ])

        # MULTI-SHELL
        if nbUniqueBvals>2:
                print('multi-shell')
                preproc.connect([
                        # FREE WATER COMPUTATION
                        (eddy, fwDWI, [('out_corrected', 'in_file')]),
                        (infoSource, fwDWI, [('bval', 'in_bval')]),
                        (eddy, fwDWI, [('out_rotated_bvecs', 'in_bvec')]),
                        (rename, fwDWI, [('out_file', 'in_mask')]),


                        # INTRAVOXEL INCOHERENT MOTION
                        #(eddy, ivim, [('out_corrected', 'in_file')]),
                        #(infoSource, ivim, [('bval', 'in_bval')]),
                        #(eddy, ivim, [('out_rotated_bvecs', 'in_bvec')]),

                        # FREEWATER DATA SINK
                        (fwDWI, dataSink, [('out_FA', 'DWIspace.fwe.@fweFA')]),
                        (fwDWI, dataSink, [('out_MD', 'DWIspace.fwe.@fweMD')]),
                        (fwDWI, dataSink, [('out_FW', 'DWIspace.fwe.@fweFW')]),
                        (fwDWI, dataSink, [('out_eVecs', 'DWIspace.fwe.@fweVecs')]),
                        (fwDWI, dataSink, [('out_eVals', 'DWIspace.fwe.@fweVals')]),
                        (fwDWI, dataSink, [('out_FA_thresholded', 'DWIspace.fwe.@fweFAthr')]),
                        (fwDWI, dataSink, [('out_fwmask', 'DWIspace.fwe.@fweMask')]),
                ])

        return preproc


if __name__=='__main__':
        #print(colours.green + 'Build Pipeline.' + colours.ENDC)
        pipeLine = BuildPipeLine()
        pipeLine.write_graph()
        #print(colours.green + 'Run Pipeline...' + colours.ENDC)
        pipeLine.run(plugin='MultiProc')
        #print(colours.green + 'Pipeline completed.' + colours.ENDC)

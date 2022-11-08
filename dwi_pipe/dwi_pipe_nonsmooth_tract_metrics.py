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
                opts, args = getopt.getopt(sys.argv[1:],'-ho:c:d:b:r:e:k:p:m:', ['help', 'outDir=','name=','DWI=','bval=','bvec=', 'DWIMask=', 'smooth=','prob_atlas=','bin_atlas='])
        except getopt.GetoptError as err:
                print(err) # will print(something like "option <> not recognized"
                print('usage: diffusion_pipeline.py',
                        '\n -o <output directory>',
                        '\n -c <name_of_nipype>',
                        '\n -d <directory/to/DWI.nii>',
                        '\n -b <directory/to/DWI.bval>',
                        '\n -r <directory/to/DWI.bvec>',
                        '\n -e <directory/to/DWImask.nii>',
                        '\n -k <smoothing_factor>',
                        '\n -p <directory/to/probability/atlas.nii>',
                        '\n -m <directory/to/binary/atlas.nii>')
                        #'\n -i <directory/to/index.txt>')
                sys.exit(2)

        print(opts)
        for opt, arg in opts:
                if opt in ('-h','--help') or len(opts)!=9:
                        print('usage: diffusion_pipeline.py',
                        '\n -o <output directory>',
                        '\n -c <name_of_nipype>',
                        '\n -d <directory/to/DWI.nii>',
                        '\n -b <directory/to/DWI.bval>',
                        '\n -r <directory/to/DWI.bvec>',
                        '\n -e <directory/to/DWImask.nii>',
                        '\n -k <smoothing_factor>',
                        '\n -p <directory/to/probability/atlas.nii>',
                        '\n -m <directory/to/binary/atlas.nii>')
                        sys.exit(2)

                elif opt in ('-o','--outDir'):
                        outDir = arg
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
                elif opt in ('-k','--smooth'):
                        smoothFactor = arg
                elif opt in ('-p','--prob_atlas'):
                        probAtlas = arg
                elif opt in ('-m','--bin_atlas'):
                        binAtlas = arg
                else:
                        assert False, "unhandled option"

        # check if a DWI mask was specified
        if DWImaskFile=='None':
                print('No DWImaks was provided')
                DWImaskFile = None
                outputFolderName = 'diffusion_dwi'
        else:
                outputFolderName = 'diffusion_dw_manualMask'



        #-----------------------------------------------------------------------------------------------------#
        # INPUT SOURCE NODES
        #-----------------------------------------------------------------------------------------------------#
        #print(colours.green + "Create Source Nodes." + colours.ENDC
        infoSource = pe.Node(IdentityInterface(fields=['dwi_file', 'bval', 'bvec', 'dwi_mask','smooth','prob_atlas','bin_atlas']), name='infosource')
        infoSource.inputs.dwi_file = DWIfile
        infoSource.inputs.bval = Bvalfile
        infoSource.inputs.bvec = Bvecfile
        infoSource.inputs.dwi_mask = DWImaskFile
        infoSource.inputs.smooth = smoothFactor
        infoSource.inputs.out_probability_atlas = probAtlas
        infoSource.inputs.out_binary_atlas = binAtlas

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
        txtFile = os.path.join(outDir,'inputFiles_' + outputFolderName + '_pip.txt')
        if os.path.isfile(txtFile):
                print('Old <inputFiles_' + outputFolderName + '_pip.txt> removed.')
                os.remove(txtFile)

        # check if outputdolder exists, otherwise create
        if not os.path.exists(outDir):
                os.makedirs(outDir)


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

        print('{0:8} | '.format('outDir'), outDir)
        print('-------------------')

        output = '{0:8} | '.format('outDir') + outDir + '\n-------------------\n'
        f.write(output)
        f.close()

        # ASSESS 3rd DIMENSION OF DWI IMAGE
        # topup runs with given paprameters only when dimensions are even
        # if dimesnions are odd, cut off first slice, only do so when extra b0 is provided
        # otherwise topup is not applied anyways, so no cropping will take place
        valueForCutOfAxialSlice_X = 0
        valueForCutOfAxialSlice_Y = 0
        valueForCutOfAxialSlice_Z = 0


        #-----------------------------------------------------------------------------------------------------#
        # WORKFLOW
        #-----------------------------------------------------------------------------------------------------#
        #print(colours.green + "Create Workflow" + colours.ENDC)
        preproc = pe.Workflow(name=outputFolderName)
        preproc.base_dir = outDir
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

        # HELPER NODE TO SELECT DWI or DWI_B0 file
        renameDWI =  pe.Node(Rename(), name='select_dwi')
        renameDWI.inputs.format_string = 'selectedDWI.nii.gz'

        # DENOISING
        lpca = pe.Node(mrtrix3.DWIDenoise(), name='LPCA')

        # GIBBS RINGING REMOVAL
        gibbs = pe.Node(HelperInterfaces.MRTRIX3GibbsRemoval(), name='gibbs_removal')

        # dwimask is a mask that excludes L1<0 and FA>1
        dwiMask = pe.Node(HelperInterfaces.DWIMask(), name='dwifit_mask')

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

        # WM PARCELLATION with TractSeg, version 1 (Cambridge-based)
        tractSegCam = pe.Node(HelperInterfaces.TractSegCam(), name='tract_seg_cam')

        # WM PARCELLATION WITH TractSeg, version 2
        # check the gradient directions
        #mrtrixGradCheck = pe.Node(HelperInterfaces.MRTRIX3GradCheck(), name='mrtrix_grad_check')
        # calculate the CSD and the peaks
        #createFods = pe.Node(HelperInterfaces.CreateFods(), name='create_fods')
        # flip the x-direction of the peaks 
        #flipPeaks = pe.Node(HelperInterfaces.FlipPeaks(), name='flip_peaks')
        # determine the wm parcellation with TractSeg 
        #peaksTractSeg = pe.Node(HelperInterfaces.PeaksTractSeg(), name='peaks_tractseg')
        # calculate metrics of the tracts 
        #tractMetrics = pe.Node(HelperInterfaces.TractMetrics(), name='tract_metrics')
        
        # WM PARCELLATION WITH TractSeg, version 3
        # check the gradient directions
        mrtrixGradCheck = pe.Node(HelperInterfaces.MRTRIX3GradCheck(), name='mrtrix_grad_check')
        # raw tract segmentation after mrtrix 
        tractSeg = pe.Node(HelperInterfaces.RawTractSeg(), name='tract_seg')
        # calculate metrics of the tracts 
        tractMetrics = pe.Node(HelperInterfaces.TractMetrics(), name='tract_metrics')
        # QC - mosaic of tracts 
        mosaicTracts = pe.Node(HelperInterfaces.MosaicTracts(), name='QC_tracts_mosaic')

        # SMOOTH DWI DATA
        dtismooth = pe.Node(HelperInterfaces.DTIsmooth(), name='dtismooth')
        dtismooth.inputs.in_smooth = float(infoSource.inputs.smooth)
        
        # DIFFUSION TENSOR RECONSTRUCTION
        dtifit = pe.Node(fsl.DTIFit(), name='dtifit')
        dtifit.inputs.args = '-w'
        dtifit.inputs.base_name = 'dtifitWLS'
        dtifit.inputs.output_type = 'NIFTI_GZ'
        
        # DTI-RD 
        dtird = pe.Node(HelperInterfaces.RDCompute(), name='dtird')

        # DIFFUSION KURTOSIS RECONSTRUCTION
        dkifit = pe.Node(HelperInterfaces.DKIfit(), name='dkifit')
        dkifit.inputs.in_smooth = float(infoSource.inputs.smooth)

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
        ants_warpDiffMaps.inputs.output_image = ['MD_DWI_anatSPace.nii.gz', 'FA_DWI_anatSPace.nii.gz', 'MK_DWI_anatSPace.nii.gz','AD_DWI_anatSPace.nii.gz', 'RD_DWI_anatSpace.nii.gz']
        ants_warpDiffMaps.inputs.interpolation = 'WelchWindowedSinc'

        ants_warpPM = pe.Node(ants.ApplyTransforms(), name='ants_warpPM')
        ants_warpPM.inputs.dimension = 3
        ants_warpPM.inputs.output_image = 'APM_DWI_anatSpace.nii.gz'
        ants_warpPM.inputs.interpolation = 'WelchWindowedSinc'


        ants_warpTraceMaps = pe.MapNode(ants.ApplyTransforms(), iterfield=['input_image', 'output_image'], name='ants_warpTraceMaps', synchronize=True)
        ants_warpTraceMaps.inputs.dimension = 3
        ants_warpTraceMaps.inputs.output_image = ['traceMap_b' + str(int(b)).zfill(4)  + '_DWI_anatSpace.nii.gz' for b in uniqueBvals]
        ants_warpTraceMaps.inputs.interpolation = 'WelchWindowedSinc'

        # DWITK only prepare for tensor registration
        prepDtitk = pe.Node(HelperInterfaces.PrepareDTITK(), name='dwitk_prepare')

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
        dataSink.inputs.base_directory = outDir
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

        if "dki" in nipype_name:
         preproc.connect([
                # DIFFUSION KURTOSIS
                # dkifit
                (infoSource, dkifit, [('dwi_file', 'in_file')]),
                (infoSource, dkifit, [('bval', 'in_bvals')]),
                (infoSource, dkifit, [('bvec', 'in_bvecs')]),
                (infoSource, dkifit, [('dwi_mask', 'in_mask')]),


                # tract metrics 
                (infoSource, tractMetrics, [('out_probability_atlas','in_probability_atlas')]),
                (infoSource, tractMetrics, [('out_binary_atlas', 'in_binary_atlas')]),
                (dkifit, tractMetrics, [('out_fa', 'in_fa')]),
                (dkifit, tractMetrics, [('out_md', 'in_md')]),  
		(dkifit, tractMetrics, [('out_mk', 'in_mk')]), 
		(dkifit, tractMetrics, [('out_ak', 'in_ak')]), 
		(dkifit, tractMetrics, [('out_rk', 'in_rk')]), 
		(dkifit, tractMetrics, [('out_rd', 'in_rd')]), 
		(dkifit, tractMetrics, [('out_ad', 'in_ad')]),
 
                (dkifit, dwiMask, [('out_fa', 'in_FA')]),
                (dkifit, dwiMask, [('out_ad', 'in_L1')]),
                (infoSource, QC_brainMasks, [('dwi_mask', 'in_file1')]),
                (infoSource, QC_brainMasks, [('dwi_mask', 'in_file2')]),
                (QC_brainMasks, QC_report, [('out_volume1', 'mask_volume1')]),
                (QC_brainMasks, QC_report, [('out_volume2', 'mask_volume2')]),
                (QC_brainMasks, QC_report, [('out_volumeRatio', 'mask_volumeRatio')]),
                (dwiMask, QC_report, [('out_numberOutlier', 'dtifit_outlier')]),
        # ---------------------------------------
                #  DATA SINK
                # ---------------------------------------

		# FSL DKIFIT
		(dkifit, dataSink, [('out_fa', 'DWIspace.dki.@fa_dki')]),
		(dkifit, dataSink, [('out_md', 'DWIspace.dki.@md_dki')]),
		(dkifit, dataSink, [('out_mk', 'DWIspace.dki.@mk')]),
		(dkifit, dataSink, [('out_ad', 'DWIspace.dki.@ad')]),
		(dkifit, dataSink, [('out_rd', 'DWIspace.dki.@rd')]),
		(dkifit, dataSink, [('out_ak', 'DWIspace.dki.@ak')]),
		(dkifit, dataSink, [('out_rk', 'DWIspace.dki.@rk')]),



                (tractMetrics, dataSink, [('out_csv_report', 'DWIspace.WM_parcellation.@report')]),

                # QCs
                (QC_report, dataSink, [('out_csv', 'DWIspace.QC.@report')]),

         ])
        else:
         preproc.connect([
                # DIFFUSION TENSOR RECONSTRUCTION
                # dtifit
                (infoSource, dtifit, [('dwi_file', 'dwi')]),
                (infoSource, dtifit, [('bval', 'bvals')]),
                (infoSource, dtifit, [('bvec', 'bvecs')]),
                (infoSource, dtifit, [('dwi_mask', 'mask')]),

                # tract metrics 
                (infoSource, tractMetrics, [('out_probability_atlas','in_probability_atlas')]),
                (infoSource, tractMetrics, [('out_binary_atlas', 'in_binary_atlas')]),
                (dtifit, tractMetrics, [('FA', 'in_fa')]),
                (dtifit, tractMetrics, [('MD', 'in_md')]), 
 
                (dtifit, dtird, [('L2', 'in_l2')]),  
                (dtifit, dtird, [('L3', 'in_l3')]),  
                (infoSource, dtird, [('dwi_mask', 'in_mask')]),  
                (dtifit, tractMetrics, [('L1', 'in_ad')]),  
                (dtird, tractMetrics, [('out_rd', 'in_rd')]),  
                (dtifit, tractMetrics, [('MO', 'in_mo')]),  

                (dtifit, dwiMask, [('FA', 'in_FA')]),
                (dtifit, dwiMask, [('L1', 'in_L1')]),
                (infoSource, QC_brainMasks, [('dwi_mask', 'in_file1')]),
                (infoSource, QC_brainMasks, [('dwi_mask', 'in_file2')]),
                (QC_brainMasks, QC_report, [('out_volume1', 'mask_volume1')]),
                (QC_brainMasks, QC_report, [('out_volume2', 'mask_volume2')]),
                (QC_brainMasks, QC_report, [('out_volumeRatio', 'mask_volumeRatio')]),
                (dwiMask, QC_report, [('out_numberOutlier', 'dtifit_outlier')]),

                # FSL DTIFIT
                (dtifit, dataSink, [('FA', 'DWIspace.dti.@fa')]),
                (dtifit, dataSink, [('MD', 'DWIspace.dti.@md')]),
                (dtifit, dataSink, [('MO', 'DWIspace.dti.@m0')]),
                (dtifit, dataSink, [('S0', 'DWIspace.dti.@s0')]),
                (dtifit, dataSink, [('V1', 'DWIspace.dti.@v1')]),
                (dtifit, dataSink, [('V2', 'DWIspace.dti.@v2')]),
                (dtifit, dataSink, [('V3', 'DWIspace.dti.@v3')]),
                (dtifit, dataSink, [('L1', 'DWIspace.dti.@l1')]),
                (dtifit, dataSink, [('L2', 'DWIspace.dti.@l2')]),
                (dtifit, dataSink, [('L3', 'DWIspace.dti.@l3')]),
                (dtird, dataSink, [('out_rd', 'DWIspace.dti.@rd')]),
                (tractMetrics, dataSink, [('out_csv_report', 'DWIspace.WM_parcellation.@report')]),

                # QCs
                (QC_report, dataSink, [('out_csv', 'DWIspace.QC.@report')]),


        ])

        return preproc


if __name__=='__main__':
        #print(colours.green + 'Build Pipeline.' + colours.ENDC)
        pipeLine = BuildPipeLine()
        pipeLine.write_graph()
        #print(colours.green + 'Run Pipeline...' + colours.ENDC)
        pipeLine.run(plugin='MultiProc')
        #print(colours.green + 'Pipeline completed.' + colours.ENDC)

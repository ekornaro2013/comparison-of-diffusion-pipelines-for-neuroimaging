# The current script was developped by:
# Evgenios N. Kornaropoulos - evgenios.kornaropoulos@med.lu.se
# and
# Stefan Winzeck - sw742@cam.ac.uk
# 02/2019
#
# For co-author list:
# Stefan Winzeck, Division of Anaesthesia, Department of Medicine, University of Cambridge, UK
# Marta M. Correia, MRC Cognition and Brain Sciences Unit, University of Cambridge, Cambridge, UK
#

import sys, os, copy, shutil, csv
import nibabel as nib
import numpy as np

from itertools import product
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec,
    traits, TraitedSpec, File, isdefined,
    CommandLine, CommandLineInputSpec
)

from nipype.utils.filemanip import split_filename
from dipy.io import read_bvals_bvecs
from dipy.data import gradient_table
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.denoise.localpca import localpca

from pandas import DataFrame as df
import dipy.reconst.fwdti as fwdti
from dipy.reconst.ivim import IvimModel
from scipy import stats
from scipy import signal

#-----------------------------------------------------------------------------------------------------#
# COMPUTE AXIAL AND RADIAL DIFFUSIVITY MAPS
#-----------------------------------------------------------------------------------------------------#
class RDComputeInputSpec(BaseInterfaceInputSpec):
    in_l2 = traits.File(exists=True, desc='L2 volume', mandatory=True)
    in_l3 = traits.File(exists=True, desc='L3 volume', mandatory=True)
    in_mask = traits.File(exists=True, desc='brain mask', mandatory=True)

class RDComputeOutputSpec(TraitedSpec):
    out_rd = traits.File(exists=True, desc='RD map')

class RDCompute(BaseInterface):
    input_spec = RDComputeInputSpec
    output_spec = RDComputeOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.in_l2)
        output_dir = os.path.abspath('')
        l2 = self.inputs.in_l2
        l3 = self.inputs.in_l3
        mask = self.inputs.in_mask
        l2_load = nib.load(l2)
        l2_img = l2_load.get_data()
        l3_load = nib.load(l3)
        l3_img = l3_load.get_data()
        mask_load = nib.load(mask)
        mask_img = mask_load.get_data()
        rd_img = np.copy(mask_img)
        self.out_rd = os.path.join(output_dir, 'dtifitWLS_RD.nii.gz')
        for k,j,i in product(range(l2_img.shape[0]),range(l2_img.shape[1]),range(l2_img.shape[2])):
            if mask_img[k,j,i] > 0.0:
               rd_img[k,j,i] = (l2_img[k,j,i] + l3_img[k,j,i])/2

        RD_file = nib.Nifti1Image(rd_img, l2_load.affine, l2_load.get_header())
        nib.save(RD_file, self.out_rd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_rd'] = self.out_rd

        return outputs

#-----------------------------------------------------------------------------------------------------#
# QC report
#-----------------------------------------------------------------------------------------------------#

class QCReportInputSpec(BaseInterfaceInputSpec):
    in_SNR = traits.Float(exists=True, desc='signal to noise ratio value')
    in_NCC = traits.Float(exists=True, desc='normalised cross correlation value')
    mask_volume1 = traits.Float(mandatory=True, desc='mask volume 1')
    mask_volume2 = traits.Float(mandatory=True, desc='mask volume 2')
    mask_volumeRatio = traits.Float(mandatory=True, desc='ratio og volume1/volume2')
    eddy_avg_total_motion = traits.Float(exists=True, desc='QC output: average motion between volumes and 1st volumes')
    eddy_max_total_motion = traits.Float(exists=True, desc='QC output: maximum motion between volumes and 1st volumes')
    eddy_avg_relative_motion = traits.Float(exists=True, desc='QC output: average motion between volumes and previous volumes')
    eddy_max_relative_motion = traits.Float(exists=True, desc='QC output: maximum motion between volumes and previous volumes')
    dtifit_outlier = traits.Float(exists=True, desc='QC output: number of vocels for which dtifti failed')

class QCReportOutputSpec(TraitedSpec):
    out_csv = File(exists=True, desc='QC report in csv format')

class QCReport(BaseInterface):
    input_spec = QCReportInputSpec
    output_spec = QCReportOutputSpec

    def _run_interface(self, runtime):
        # SNR
        snr_headers = ['b0 SNR']
        snr_values = [self.inputs.in_SNR]

        # coreg comparison
        coreg_headers = ['NCC T1-FA']
        coreg_values = [self.inputs.in_NCC]

        # mask comparison
        mask_headers = ['ANTS/manual brain volume', 'MRTRIX brain volume', 'ANTS/MRTRIX volume ratio']
        mask_values = [self.inputs.mask_volume1, self.inputs.mask_volume2, self.inputs.mask_volumeRatio]

        eddy_headers = ['avg total motion', 'max total motion', 'avg relative motion', 'max relative motion']
        eddy_values = [self.inputs.eddy_avg_total_motion, self.inputs.eddy_max_total_motion, self.inputs.eddy_avg_relative_motion, self.inputs.eddy_max_relative_motion]

        outlier_headers = ['DTIFIT num outlier']
        outlier_values = [self.inputs.dtifit_outlier]

        headers = snr_headers + coreg_headers + mask_headers + eddy_headers + outlier_headers
        QC_values = snr_values + coreg_values + mask_values + eddy_values + outlier_values
        with open(os.path.abspath('DTI_QC_report.csv'), 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(headers)
            filewriter.writerow(QC_values)

        self.out_csv = os.path.abspath('DTI_QC_report.csv')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_csv'] = self.out_csv
        return outputs


#-----------------------------------------------------------------------------------------------------#
# MRTRX3 BRAIN MASK
#-----------------------------------------------------------------------------------------------------#
class MRTRIX3BrainMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='input image to mask', mandatory=True)
    in_bval = File(exists=True, desc='input bval to mask', mandatory=True)
    in_bvec = File(exists=True, desc='input bvec to mask', mandatory=True)

class MRTRIX3BrainMaskOutputSpec(TraitedSpec):
    out_mask = File(exists=True, desc='mask of input image')

class MRTRIX3BrainMask(BaseInterface):
    input_spec = MRTRIX3BrainMaskInputSpec
    output_spec = MRTRIX3BrainMaskOutputSpec

    def _run_interface(self, runtime):
        # define output folders
        _, base, _ = split_filename(self.inputs.in_file)
        output_dir = os.path.abspath('')

        # define names for output files
        self.out_mask = os.path.join(output_dir, 'MRtrix3_brain_mask.nii.gz')

        gradFiles = ' '.join((self.inputs.in_bvec, self.inputs.in_bval))
        # initialise and run ROBEX
        _brainmask = BRAINMASK(in_grad=gradFiles, input_file=self.inputs.in_file, output_file=self.out_mask)
        _brainmask.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_mask'] = self.out_mask
        return outputs

    def _gen_filename(self, name):
        if name == 'out_mask':
            return self._gen_outfilename()
        return None

class BRAINMASKInputSpec(CommandLineInputSpec):
    in_grad = traits.Str(exists=True, desc='input file to be corrected for rinning artefacts',
                            mandatory=True, argstr="-fslgrad %s", position=0)
    input_file = traits.Str(exists=True, desc='input file to mask',
                            mandatory=True, argstr="%s", position=1)
    output_file = traits.Str(desc='out mask file',
                            mandatory=True, argstr="%s", position=2)

class BRAINMASK(CommandLine):
    input_spec = BRAINMASKInputSpec
    _cmd = 'dwi2mask -force'


#-----------------------------------------------------------------------------------------------------#
# MRTRIX3 GIBBS RINGING REMOVAL
#-----------------------------------------------------------------------------------------------------#
class MRTRIX3GibbsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='input image to correct for Gibbs ringing', mandatory=True)
    #axes = List(traits.Int, desc='list select the slice axes (default: 0,1 - i.e. x-y)')
    #nshifts = traits.Int(desc='value discretization of subpixel spacing (default: 20)')
    #minW = traits.Int(desc='value left border of window used for TV computation (default: 1)')
    #maxW = traits.Int(desc='value right border of window used for TV computation (default: 3)')
    # TODO add more arguments

class MRTRIX3GibbsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='image corrected for Gibbs ringing artefacts')

class MRTRIX3GibbsRemoval(BaseInterface):
    input_spec = MRTRIX3GibbsInputSpec
    output_spec = MRTRIX3GibbsOutputSpec

    def _run_interface(self, runtime):
        # define output folders
        _, base, _ = split_filename(self.inputs.in_file)
        output_dir = os.path.abspath('')

        # define names for output files
        self.corrected_output = os.path.join(output_dir, base + '_gibbs.nii.gz')

        # initialise and run ROBEX
        _gibbs = GIBBS(input_file=self.inputs.in_file, output_file=self.corrected_output)
        _gibbs.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.corrected_output
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None

class GIBBSInputSpec(CommandLineInputSpec):
    input_file = traits.Str(exists=True, desc='input file to be corrected for rinning artefacts',
                            mandatory=True, argstr="%s", position=0)
    output_file = traits.Str(desc='corrected file',
                            mandatory=True, argstr="%s", position=1)

class GIBBS(CommandLine):
    input_spec = GIBBSInputSpec
    _cmd = 'mrdegibbs -force'


#-----------------------------------------------------------------------------------------------------#
# COMPARE BRAIN MASKS
#-----------------------------------------------------------------------------------------------------#
class QCcompareBrainMasksInputSpec(BaseInterfaceInputSpec):
    in_file1 = traits.File(exists=True, desc='mask volume 1', mandatory=True)
    in_file2 = traits.File(exists=True, desc='mask volume 2', mandatory=True)

class QCcompareBrainMasksOutputSpec(TraitedSpec):
    out_volume1 = traits.Float(exists=True, desc='QC output: comparision of brain mask')
    out_volume2 = traits.Float(exists=True, desc='QC output: comparision of brain mask')
    out_volumeRatio = traits.Float(exists=True, desc='QC output: comparision of brain mask')

class QCcompareBrainMasks(BaseInterface):
    input_spec = QCcompareBrainMasksInputSpec
    output_spec = QCcompareBrainMasksOutputSpec

    def _run_interface(self, runtime):
        volumes = []
        for in_file in [self.inputs.in_file1, self.inputs.in_file2]:
            imgFile = nib.load(in_file)
            img = imgFile.get_data()
            voxlVolume = np.prod(imgFile.header.get_zooms())
            maskVolume = np.sum(img) * voxlVolume / 1000.0  #volume in ml
            volumes.append(maskVolume)

        self.out_volume1 = volumes[0]
        self.out_volume2 = volumes[1]
        self.volumeRatio = self.out_volume2 / self.out_volume1 * 100.0

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_volume1'] = self.out_volume1
        outputs['out_volume2'] = self.out_volume2
        outputs['out_volumeRatio'] = self.volumeRatio

        return outputs

#-----------------------------------------------------------------------------------------------------#
# COMPUTE NORMALISED CROSS CORRELATION
#-----------------------------------------------------------------------------------------------------#
class ComputeNCCInputSpec(BaseInterfaceInputSpec):
    in_file1 = traits.File(exists=True, desc='volume 1', mandatory=True)
    in_file2 = traits.File(exists=True, desc='volume 2', mandatory=True)
    in_mask = traits.File(exists=True, desc='mask volume to compute NCC in', mandatory=True)

class ComputeNCCOutputSpec(TraitedSpec):
    NCC = traits.Float(exists=True, desc='QC output: normalised cross correlation')

class ComputeNCC(BaseInterface):
    input_spec = ComputeNCCInputSpec
    output_spec = ComputeNCCOutputSpec

    def _run_interface(self, runtime):
        img1 = np.squeeze(nib.load(self.inputs.in_file1).get_data())
        img2 = np.squeeze(nib.load(self.inputs.in_file2).get_data())

        if isdefined(self.inputs.in_mask):
            mask = np.squeeze(nib.load(self.inputs.in_mask).get_data())
        else:
            mask = np.ones(img1.shape)

        ind = np.where(mask==1)

        mean1 = np.mean(img1[ind])
        mean2 = np.mean(img2[ind])
        std1 = np.std(img1[ind])
        std2 = np.std(img2[ind])
        tmp = (img1 - mean1) * (img2 - mean2)
        self.NCC = np.mean(tmp[ind]) / (std1 * std2)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['NCC'] = self.NCC

        return outputs


#-----------------------------------------------------------------------------------------------------#
# COMPUTE SIGNAL TO NOISE RATIO
#-----------------------------------------------------------------------------------------------------#
class ComputeSNRInputSpec(BaseInterfaceInputSpec):
    in_signal = traits.File(exists=True, desc='volume 1', mandatory=True)
    in_noise = traits.File(exists=True, desc='volume 2', mandatory=True)
    in_mask = traits.File(exists=True, desc='mask volume to compute SNR in', mandatory=True)

class ComputeSNROutputSpec(TraitedSpec):
    SNR = traits.Float(exists=True, desc='QC output: signal to noise ratio')

class ComputeSNR(BaseInterface):
    input_spec = ComputeSNRInputSpec
    output_spec = ComputeSNROutputSpec

    def _run_interface(self, runtime):
        signal = nib.load(self.inputs.in_signal).get_data()[:,:,:,0]
        noise = nib.load(self.inputs.in_noise).get_data()[:,:,:,0]

        if isdefined(self.inputs.in_mask):
            mask = nib.load(self.inputs.in_mask).get_data()
        else:
            mask = np.ones(img1.shape)

        ind = np.where(mask==1)

        mean_signal = np.mean(signal[ind])
        std_noise = np.std(noise[ind])
        self.SNR = mean_signal / std_noise

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['SNR'] = self.SNR

        return outputs


#-----------------------------------------------------------------------------------------------------#
# EDDY QC
#-----------------------------------------------------------------------------------------------------#
class EddyQCInputSpec(BaseInterfaceInputSpec):
    in_eddy_rms = File(exists=True, desc='output rms file from eddy', mandatory=True)

class EddyQCOutputSpec(TraitedSpec):
    out_avg_total_motion = traits.Float(exists=True, desc='QC output: average motion between volumes and 1st volumes')
    out_max_total_motion = traits.Float(exists=True, desc='QC output: maximum motion between volumes and 1st volumes')
    out_avg_relative_motion = traits.Float(exists=True, desc='QC output: average motion between volumes and previous volumes')
    out_max_relative_motion = traits.Float(exists=True, desc='QC output: maximum motion between volumes and previous volumes')

class EddyQC(BaseInterface):
    input_spec = EddyQCInputSpec
    output_spec = EddyQCOutputSpec

    def _run_interface(self, runtime):
        # define output folders
        eddy_rms = self._readTxtFile(self.inputs.in_eddy_rms)

        # compute stats on motion parameters
        # exclude first line as it is first volume (no motion to itself)
        self.avg_total_motion = np.mean(eddy_rms[1:,:], axis=0)[0]
        self.max_total_motion = np.max(eddy_rms[1:,:], axis=0)[0]
        self.avg_relative_motion = np.mean(eddy_rms[1:,:], axis=0)[1]
        self.max_relative_motion = np.max(eddy_rms[1:,:], axis=0)[1]

        return runtime


    def _readTxtFile(self, fileName):
        with open(fileName) as f:
            content = f.readlines()

        content = [x.strip() for x in content] # strips \n and blanks
        rms = np.vstack([(c.split(' ')[0], c.split(' ')[2]) for c in content]).astype(np.float32)

        return rms


    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_avg_total_motion'] = self.avg_total_motion
        outputs['out_max_total_motion'] = self.max_total_motion
        outputs['out_avg_relative_motion'] = self.avg_relative_motion
        outputs['out_max_relative_motion'] = self.max_relative_motion

        return outputs


#-----------------------------------------------------------------------------------------------------#
# TRACT SEG
#-----------------------------------------------------------------------------------------------------#
class TractSegCamInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='DWI volume to be parcellated', mandatory=True)
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(desc='brain mask to calculate tracts in')
    in_fa = File(desc='FA maps to compute metrics on')
    in_md = File(desc='MD maps to compute metrics on')

class TractSegCamOutputSpec(TraitedSpec):
    out_probability_atlas = File(desc='tract probability atlas 4D')
    out_segmentation_atlas = File(desc='tract segmentation atlas 4D')
    out_peaks = File(desc='peaks')    
    out_csv_report = File(desc='csv file with ROI volumes and intensities')

class TractSegCam(BaseInterface):
    input_spec = TractSegCamInputSpec
    output_spec = TractSegCamOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.in_mask):
            maskInput = "--brain_mask " + self.inputs.in_mask
        else:
            maskInput = ""

        # define output folders
        _, base, _ = split_filename(self.inputs.in_file)
        folder_output_dir = os.path.abspath('')
        # output_dir = os.path.join(folder_output_dir, 'tractseg_output')
        output_dir = os.path.join(folder_output_dir) 
        
        # define names for output files
        self.probability_atlas = os.path.join(output_dir, 'bundle_segmentations.nii.gz')
        self.binary_atlas = os.path.join(output_dir, 'bundle_segmentations_binary.nii.gz')
        self.peaks = os.path.join(output_dir, 'peaks.nii.gz')
        self.out_csv_name = os.path.join(output_dir, 'TractSeg_report.csv')

        # initialise and run TractSeg
        _tractseg = TRAGTSEG(input_file=self.inputs.in_file,
            input_bvals=self.inputs.in_bvals,
            input_bvecs=self.inputs.in_bvecs,
            input_mask=maskInput,
            output_dir=folder_output_dir)
        _tractseg.run()

        atlas_proxy = nib.load(self.probability_atlas)
        probability_atlas = np.squeeze(atlas_proxy.get_data())
        numberOfROIs = probability_atlas.shape[3] # 4th dimension of 4D atlas equates number of ROIs

        # create binary atals by thresholding probabilty atlas at 0.5
        segmentation_atlas = np.where(probability_atlas>0.5, 1, 0)
        segmentation_atlasFile = nib.Nifti1Image(segmentation_atlas, atlas_proxy.affine, atlas_proxy.get_header())
        nib.save(segmentation_atlasFile, self.binary_atlas)

        # check if additional files were provided and load if available
        if isdefined(self.inputs.in_fa):
            fa = nib.load(self.inputs.in_fa).get_data()
            metricInROI_mad_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_fa = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            fa = None
            metricInROI_mad_fa = ['n/a'] * numberOfROIs
            metricInROI_median_fa = ['n/a'] * numberOfROIs
            metricInROI_mean_fa = ['n/a'] * numberOfROIs
            metricInROI_std_fa = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_fa = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_fa = ['n/a'] * numberOfROIs

        if isdefined(self.inputs.in_md):
            md = nib.load(self.inputs.in_md).get_data()
            metricInROI_mad_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_md = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            md = None
            metricInROI_mad_md = ['n/a'] * numberOfROIs
            metricInROI_median_md = ['n/a'] * numberOfROIs
            metricInROI_mean_md = ['n/a'] * numberOfROIs
            metricInROI_std_md = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_md = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_md = ['n/a'] * numberOfROIs

        # compute ROI IDPs
        voxelVolume_mm3 = np.prod(atlas_proxy.get_header().get_zooms()[:3])
        ROI_volume_mm3 = np.zeros(numberOfROIs, dtype=np.float32)
        for r in range(numberOfROIs):
            roi = segmentation_atlas[:,:,:,r]
            roi_probabilities = probability_atlas[:,:,:,r]
            probabilities = roi_probabilities[roi==1]

            # compute ROI volumes
            ROI_volume_mm3[r] = np.sum(roi)*voxelVolume_mm3

            # compute (weighted) ROI median for fa
            if type(fa)==np.ndarray:
                tmp_fa = fa[roi==1]
                metricInROI_mad_fa[r] = stats.median_absolute_deviation(tmp_fa)
                metricInROI_median_fa[r] = np.median(tmp_fa)
                metricInROI_mean_fa[r] = np.mean(tmp_fa)
                metricInROI_std_fa[r] = np.std(tmp_fa)
                metricInROI_weightedMean_fa[r] = np.sum(tmp_fa * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_fa[r] = np.sqrt(np.mean((tmp_fa - metricInROI_weightedMean_fa[r])**2))

            # compute (weighted) ROI median for md
            if type(md)==np.ndarray:
                tmp_md = md[roi==1]
                metricInROI_mad_md[r] = stats.median_absolute_deviation(1000*tmp_md)
                metricInROI_median_md[r] = np.median(1000*tmp_md)
                metricInROI_mean_md[r] = np.mean(1000*tmp_md)
                metricInROI_std_md[r] = np.std(1000*tmp_md)
                metricInROI_weightedMean_md[r] = np.sum(1000*tmp_md * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_md[r] = np.sqrt(np.mean((1000*tmp_md - metricInROI_weightedMean_md[r])**2))

        # save metrics in csv file
        with open(self.out_csv_name, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(np.hstack(['metric', np.arange(numberOfROIs)]))
            filewriter.writerow(np.hstack(['ROI_Volumes_mm3', ROI_volume_mm3]))
            filewriter.writerow(np.hstack(['ROI_Mad_FA', metricInROI_mad_fa]))
            filewriter.writerow(np.hstack(['ROI_Median_FA', metricInROI_median_fa]))
            filewriter.writerow(np.hstack(['ROI_Mean_FA', metricInROI_mean_fa]))
            filewriter.writerow(np.hstack(['ROI_Std_FA', metricInROI_std_fa]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_FA', metricInROI_weightedMean_fa]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_FA', metricInROI_weightedStd_fa]))
            filewriter.writerow(np.hstack(['ROI_Mad_MD', metricInROI_mad_md]))
            filewriter.writerow(np.hstack(['ROI_Median_MD', metricInROI_median_md]))
            filewriter.writerow(np.hstack(['ROI_Mean_MD', metricInROI_mean_md]))
            filewriter.writerow(np.hstack(['ROI_Std_MD', metricInROI_std_md]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_MD', metricInROI_weightedMean_md]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_MD', metricInROI_weightedStd_md]))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_probability_atlas'] = self.probability_atlas
        outputs['out_segmentation_atlas'] = self.binary_atlas
        outputs['out_peaks'] = self.peaks
        outputs['out_csv_report'] = self.out_csv_name

        return outputs

    def _gen_filename(self, name):
        if name == 'out_probability_atlas':
            return self._gen_outfilename()
        return None

class TRAGTSEGinputSpec(CommandLineInputSpec):
    input_file = traits.Str(exists=True, desc='input file to be masked', mandatory=True, argstr="-i %s", position=0)
    input_bvals = traits.Str(exists=True, desc='input bval file', mandatory=True, argstr="--bvals %s", position=1)
    input_bvecs = traits.Str(exists=True, desc='input bvec file', mandatory=True, argstr="--bvecs %s", position=2)
    input_mask = traits.Str(desc='input mask', mandatory=True, argstr="%s", position=3)
    output_dir = traits.Str(desc='output directory', mandatory=True, argstr="-o %s", position=4)

class TRAGTSEG(CommandLine):
    input_spec = TRAGTSEGinputSpec
    _cmd = 'python /home/fuji/Software/tractseg/TractSeg/bin/TractSeg --raw_diffusion_input --get_probabilities --single_output_file'


#-----------------------------------------------------------------------------------------------------#
# MRTRIX3 DWI GRADIENT CHECK 
#-----------------------------------------------------------------------------------------------------#
class MRTRIX3GradCheckInputSpec(BaseInterfaceInputSpec): 
    in_file = File(exists=True, desc='DWI volume to be analysed', mandatory=True)
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)

class MRTRIX3GradCheckOutputSpec(TraitedSpec): 
    out_bvecs = File(desc='gradient direction checked b-vectors')
    out_bvals = File(desc='corresponding b-values')

class MRTRIX3GradCheck(BaseInterface): 
    input_spec = MRTRIX3GradCheckInputSpec
    output_spec = MRTRIX3GradCheckOutputSpec

    def _run_interface(self, runtime): 
        # define output folder
        _, base, _ = split_filename(self.inputs.in_file)
        output_dir = os.path.abspath('')

        # define names for output files
        self.out_bvecs = os.path.join(output_dir, 'DTI_gradcheck.bvecs')    
        self.out_bvals = os.path.join(output_dir, 'DTI_gradcheck.bvals')
        
        in_gradFiles = ' '.join((self.inputs.in_bvecs, self.inputs.in_bvals))
        out_gradFiles = ' '.join((self.out_bvecs, self.out_bvals))

        # initialise and run ROBEX
        _gradcheck = DWIGRADCHECK(input_file=self.inputs.in_file, in_grad=in_gradFiles, 
                out_grad=out_gradFiles)
        _gradcheck.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_bvecs'] = self.out_bvecs
        outputs['out_bvals'] = self.out_bvals

        return outputs

    def _gen_filename(self, name):
        if name == 'out_bvecs':
            return self._gen_outfilename()
        return None

class DWIGRADCHECKinputSpec(CommandLineInputSpec):
    input_file = traits.Str(exists=True, desc='input file', mandatory=True, argstr="%s", position=0)
    in_grad = traits.Str(exists=True, desc='input bvecs and bvals', mandatory=True, argstr="-fslgrad %s", position=1)
    out_grad = traits.Str(desc='output bvecs and bvals', mandatory=True, argstr="-export_grad_fsl %s", position=4)

class DWIGRADCHECK(CommandLine):
    input_spec = DWIGRADCHECKinputSpec
    _cmd = 'python /home/fuji/Software/mrtrix3/mrtrix3/bin/dwigradcheck -number 10000 -nthreads 6'

#-----------------------------------------------------------------------------------------------------#
# RAW TO SEGMENTATION
#-----------------------------------------------------------------------------------------------------#

class RawTractSegInputSpec(BaseInterfaceInputSpec):
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(desc='brain mask to calculate tracts in')
    in_file = File(exists=True, desc='DWI image', mandatory=True)

class RawTractSegOutputSpec(TraitedSpec):
    out_probability_atlas = File(desc='tract probability atlas 4D')
    out_binary_atlas = File(desc='tract segmentation  atlas 4D')
    out_peaks = File(desc='peaks')

class RawTractSeg(BaseInterface):
    input_spec = RawTractSegInputSpec
    output_spec = RawTractSegOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.in_mask):
            maskInput = "--brain_mask " + self.inputs.in_mask
        else:
            maskInput = ""

        # define output folders
        output_dir = os.path.abspath('') 
        
        # define names for output files
        self.probability_atlas = os.path.join(output_dir, 'bundle_segmentations.nii.gz')
        self.binary_atlas = os.path.join(output_dir, 'bundle_segmentations_binary.nii.gz')
        self.peaks = os.path.join(output_dir, 'peaks.nii.gz')

        # initialise and run ROBEX
        _tractseg = TRACTSEG(input_bvals=self.inputs.in_bvals, 
                input_bvecs=self.inputs.in_bvecs,
                input_file=self.inputs.in_file,
                input_mask=maskInput,
                output_dir=output_dir)
        _tractseg.run()
        
        # save probabilty atlas 
        atlas_proxy = nib.load(self.probability_atlas)
        probability_atlas = np.squeeze(atlas_proxy.get_data())
        
        # create binary atlas by thresholding the probability atlas, 
        binary_temp = np.where(probability_atlas >0.5, 1, 0)
        
        # postprocessing is done in the same way as it is done in TractSeg 
        from tractseg.libs import img_utils
        from tractseg.data import dataset_specific_utils
        
        binary_temp = img_utils.bundle_specific_postprocessing(binary_temp, 
                dataset_specific_utils.get_bundle_names("All")[1:])
        binary_atlas = img_utils.postprocess_segmentations(binary_temp, 
                dataset_specific_utils.get_bundle_names("All")[1:], blob_thr=50, hole_closing=None)
        
        # binary segmentation is saved
        binary_atlasFile = nib.Nifti1Image(binary_atlas, atlas_proxy.affine, atlas_proxy.get_header())
        nib.save(binary_atlasFile, self.binary_atlas)
        
        return runtime
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_probability_atlas'] = self.probability_atlas
        outputs['out_binary_atlas'] = self.binary_atlas
        outputs['out_peaks'] = self.peaks
        return outputs

    def _gen_filename(self, name):
        if name == 'out_probability_atlas':
            return self._gen_outfilename()
        return None

class TRACTSEGinputSpec(CommandLineInputSpec):
    input_file = traits.Str(exists=True, desc='input file', mandatory=True, argstr="-i %s", position=0)
    input_bvals = traits.Str(exists=True, desc='input bvals', mandatory=True, argstr="--bvals %s", position=1)
    input_bvecs = traits.Str(exists=True, desc='input bvecs', mandatory=True, argstr="--bvecs %s", position=2)
    input_mask = traits.Str(exists=True, desc='input mask', mandatory=True, argstr="%s", position=3)
    output_dir = traits.Str(desc='output directory', mandatory=True, argstr="-o %s", position=4)
    
class TRACTSEG(CommandLine):
    input_spec = TRACTSEGinputSpec
    _cmd = 'python /home/fuji/Software/tractseg/TractSeg/bin/TractSeg --get_probabilities --single_output_file --raw_diffusion_input'
#-----------------------------------------------------------------------------------------------------#
# QC - MOSAIC IMAGE OF TRACTS
#-----------------------------------------------------------------------------------------------------#
class MosaicTractsInputSpec(BaseInterfaceInputSpec):
    in_binary_atlas = File(exists=True, desc='binary atlas of tracts', mandatory=True)
    in_probability_atlas = File(exists=True, desc='probability atlas of tracts', mandatory=True)
    in_file = File(exists=True, desc='DWI image', mandatory=True)

class MosaicTractsOutputSpec(TraitedSpec): 
    out_image = File(desc='mosaic file with 2d-projections of all tracts')

class MosaicTracts(BaseInterface):
    input_spec = MosaicTractsInputSpec
    output_spec = MosaicTractsOutputSpec

    def _run_interface(self, runtime): 
        # define output folders
        output_dir = os.path.abspath('') 
        
        # define name for output file
        self.image = os.path.join(output_dir, 'mosaic_tracts.png')
        
        # load the atlases 
        prob_atlas_proxy = nib.load(self.inputs.in_probability_atlas)
        probability_atlas = np.squeeze(prob_atlas_proxy.get_fdata())
        numberOfROIs = probability_atlas.shape[3]
        
        bin_atlas_proxy = nib.load(self.inputs.in_binary_atlas)
        binary_atlas = np.squeeze(bin_atlas_proxy.get_fdata())
        
        # create mosaic file with 2D-projections of all tracts in coronal view
        from tractseg.data.dataset_specific_utils import get_bundle_names
        import math 
        import matplotlib.pyplot as plt
        
        background_img = nib.load(self.inputs.in_file).get_fdata()
        background_img = background_img[:,:,:,0] 
        plt.ioff() # Turn the interactive mode off
        plt.style.use('dark_background')
        
        bundles = get_bundle_names("All")[1:]
        cols = 10
        rows = math.ceil(len(bundles)/cols)
        
        def plot_single_tract(bg, data, orientation, bundle):
        
            if orientation == "coronal":
                data = data.transpose(2, 0, 1)
                data = data[::-1, :, :]
                bg = bg.transpose(2, 0, 1)[::-1, :, :]
            elif orientation == "sagittal":
                data = data.transpose(2, 1, 0)
                data = data[::-1, :, :]
                bg = bg.transpose(2, 1, 0)[::-1, :, :]
            else:  # axial
                pass
        
            # Determines the center slice of the tract
            mask_voxel_coords = np.where(data != 0)
            if len(mask_voxel_coords) > 2 and len(mask_voxel_coords[2]) > 0:
                minidx = int(np.min(mask_voxel_coords[2]))
                maxidx = int(np.max(mask_voxel_coords[2])) + 1
                mean_slice = int(np.mean([minidx, maxidx]))
            else:
                mean_slice = int(bg.shape[2] / 2)
            bg = bg[:, :, mean_slice]
        
            # project 3D to 2D image
            data = data.max(axis=2)
        
            font = {'color':'white', 'size':4, 'verticalalignment' : 'baseline' }
            plt.imshow(bg, cmap="gray")
            data = np.ma.masked_where(data < 0.00001, data)
            plt.text(0,0, bundle, fontdict=font)
            plt.imshow(data, cmap="autumn")
        
        for j, bundle in enumerate(bundles): 
            orientation = "coronal" # could change this for the different tracts
            bundle_idx = get_bundle_names("All")[1:].index(bundle)
            mask_data = binary_atlas[:, :, :, bundle_idx]
            mask_data = np.copy(mask_data)  # copy data otherwise will also threshold data outside of plot function
        
            plt.subplot(rows, cols, j+1)
        
            plt.axis("off")
            plt.tight_layout()
            plot_single_tract(background_img, mask_data, orientation, bundle) 
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("mosaic_tracts.png", bbox_inches='tight', dpi=300)
        
        return runtime
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_image'] = self.image
        return outputs

    def _gen_filename(self, name):
        if name == 'out_image':
            return self._gen_outfilename()
        return None
#-----------------------------------------------------------------------------------------------------#
# COMPUTE METRICS OF TRACTS 
#-----------------------------------------------------------------------------------------------------#
class TractMetricsInputSpec(BaseInterfaceInputSpec): 
    in_binary_atlas = File(exists=True, desc='binary atlas of tracts', mandatory=True)
    in_probability_atlas = File(exists=True, desc='probability atlas of tracts', mandatory=True)
    in_fa = File(desc='FA maps to compute metrics on')
    in_md = File(desc='MD maps to compute metrics on')
    in_mk = File(desc='MK maps to compute metrics on')
    in_rd = File(desc='RD maps to compute metrics on')
    in_ad = File(desc='AD maps to compute metrics on')
    in_rk = File(desc='RK maps to compute metrics on')
    in_ak = File(desc='AK maps to compute metrics on')
    in_mo = File(desc='MO maps to compute metrics on')

class TractMetricsOutputSpec(TraitedSpec): 
    out_csv_report = File(desc='csv file with metrics of the tracts')

class TractMetrics(BaseInterface):
    input_spec = TractMetricsInputSpec
    output_spec = TractMetricsOutputSpec

    def _run_interface(self, runtime): 
        # define name for output file 
        self.out_csv_name = os.path.join(os.path.abspath(''), 'TractSeg_report.csv')
        
        # load the atlases 
        prob_atlas_proxy = nib.load(self.inputs.in_probability_atlas)
        probability_atlas = np.squeeze(prob_atlas_proxy.get_fdata())
        numberOfROIs = probability_atlas.shape[3]
        
        bin_atlas_proxy = nib.load(self.inputs.in_binary_atlas)

        binary_atlas = np.squeeze(bin_atlas_proxy.get_fdata())

        # check if FA and MD were provided and load if available
        if isdefined(self.inputs.in_fa):
            fa = nib.load(self.inputs.in_fa).get_data()
            metricInROI_mad_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_fa = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_fa = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            fa = None
            metricInROI_mad_fa = ['n/a'] * numberOfROIs
            metricInROI_median_fa = ['n/a'] * numberOfROIs
            metricInROI_mean_fa = ['n/a'] * numberOfROIs
            metricInROI_std_fa = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_fa = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_fa = ['n/a'] * numberOfROIs

        if isdefined(self.inputs.in_md):
            md = nib.load(self.inputs.in_md).get_data()
            metricInROI_mad_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_md = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_md = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            md = None
            metricInROI_mad_md = ['n/a'] * numberOfROIs
            metricInROI_median_md = ['n/a'] * numberOfROIs
            metricInROI_mean_md = ['n/a'] * numberOfROIs
            metricInROI_std_md = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_md = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_md = ['n/a'] * numberOfROIs

        if isdefined(self.inputs.in_mk):
            mk = nib.load(self.inputs.in_mk).get_data()
            metricInROI_mad_mk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_mk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_mk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_mk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_mk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_mk = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            mk = None
            metricInROI_mad_mk = ['n/a'] * numberOfROIs
            metricInROI_median_mk = ['n/a'] * numberOfROIs
            metricInROI_mean_mk = ['n/a'] * numberOfROIs
            metricInROI_std_mk = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_mk = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_mk = ['n/a'] * numberOfROIs

        if isdefined(self.inputs.in_rd):
            rd = nib.load(self.inputs.in_rd).get_data()
            metricInROI_mad_rd = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_rd = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_rd = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_rd = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_rd = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_rd = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            rd = None
            metricInROI_mad_rd = ['n/a'] * numberOfROIs
            metricInROI_median_rd = ['n/a'] * numberOfROIs
            metricInROI_mean_rd = ['n/a'] * numberOfROIs
            metricInROI_std_rd = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_rd = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_rd = ['n/a'] * numberOfROIs

        if isdefined(self.inputs.in_ad):
            ad = nib.load(self.inputs.in_ad).get_data()
            metricInROI_mad_ad = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_ad = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_ad = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_ad = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_ad = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_ad = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            ad = None
            metricInROI_mad_ad = ['n/a'] * numberOfROIs
            metricInROI_median_ad = ['n/a'] * numberOfROIs
            metricInROI_mean_ad = ['n/a'] * numberOfROIs
            metricInROI_std_ad = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_ad = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_ad = ['n/a'] * numberOfROIs

        if isdefined(self.inputs.in_ak):
            ak = nib.load(self.inputs.in_ak).get_data()
            metricInROI_mad_ak = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_ak = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_ak = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_ak = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_ak = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_ak = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            ak = None
            metricInROI_mad_ak = ['n/a'] * numberOfROIs
            metricInROI_median_ak = ['n/a'] * numberOfROIs
            metricInROI_mean_ak = ['n/a'] * numberOfROIs
            metricInROI_std_ak = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_ak = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_ak = ['n/a'] * numberOfROIs

        if isdefined(self.inputs.in_rk):
            rk = nib.load(self.inputs.in_rk).get_data()
            metricInROI_mad_rk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_rk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_rk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_rk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_rk = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_rk = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            rk = None
            metricInROI_mad_rk = ['n/a'] * numberOfROIs
            metricInROI_median_rk = ['n/a'] * numberOfROIs
            metricInROI_mean_rk = ['n/a'] * numberOfROIs
            metricInROI_std_rk = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_rk = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_rk = ['n/a'] * numberOfROIs

        if isdefined(self.inputs.in_mo):
            mo = nib.load(self.inputs.in_mo).get_data()
            metricInROI_mad_mo = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_median_mo = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_mean_mo = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_std_mo = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedMean_mo = np.zeros(numberOfROIs, dtype=np.float32)
            metricInROI_weightedStd_mo = np.zeros(numberOfROIs, dtype=np.float32)
        else:
            mo = None
            metricInROI_mad_mo = ['n/a'] * numberOfROIs
            metricInROI_median_mo = ['n/a'] * numberOfROIs
            metricInROI_mean_mo = ['n/a'] * numberOfROIs
            metricInROI_std_mo = ['n/a'] * numberOfROIs
            metricInROI_weightedMean_mo = ['n/a'] * numberOfROIs
            metricInROI_weightedStd_mo = ['n/a'] * numberOfROIs

        # compute ROI IDPs
        voxelVolume_mm3 = np.prod(prob_atlas_proxy.get_header().get_zooms()[:3])
        ROI_volume_mm3 = np.zeros(numberOfROIs, dtype=np.float32)
        for r in range(numberOfROIs):
            roi = binary_atlas[:,:,:,r]
            roi_probabilities = probability_atlas[:,:,:,r]
            probabilities = roi_probabilities[roi==1]

            # compute ROI volumes
            ROI_volume_mm3[r] = np.sum(roi)*voxelVolume_mm3

            # compute (weighted) ROI median for fa
            if type(fa)==np.ndarray:
                tmp_fa = fa[roi==1]
                metricInROI_mad_fa[r] = stats.median_absolute_deviation(tmp_fa)
                metricInROI_median_fa[r] = np.median(tmp_fa)
                metricInROI_mean_fa[r] = np.mean(tmp_fa)
                metricInROI_std_fa[r] = np.std(tmp_fa)
                metricInROI_weightedMean_fa[r] = np.sum(tmp_fa * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_fa[r] = np.sqrt(np.mean((tmp_fa - metricInROI_weightedMean_fa[r])**2))

            # compute (weighted) ROI median for md
            if type(md)==np.ndarray:
                tmp_md = md[roi==1]
                metricInROI_mad_md[r] = stats.median_absolute_deviation(1000*tmp_md)
                metricInROI_median_md[r] = np.median(1000*tmp_md)
                metricInROI_mean_md[r] = np.mean(1000*tmp_md)
                metricInROI_std_md[r] = np.std(1000*tmp_md)
                metricInROI_weightedMean_md[r] = np.sum(1000*tmp_md * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_md[r] = np.sqrt(np.mean((1000*tmp_md - metricInROI_weightedMean_md[r])**2))

            # compute (weighted) ROI median for mk
            if type(mk)==np.ndarray:
                tmp_mk = mk[roi==1]
                metricInROI_mad_mk[r] = stats.median_absolute_deviation(tmp_mk)
                metricInROI_median_mk[r] = np.median(tmp_mk)
                metricInROI_mean_mk[r] = np.mean(tmp_mk)
                metricInROI_std_mk[r] = np.std(tmp_mk)
                metricInROI_weightedMean_mk[r] = np.sum(tmp_mk * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_mk[r] = np.sqrt(np.mean((tmp_mk - metricInROI_weightedMean_mk[r])**2))

            if type(ad)==np.ndarray:
                tmp_ad = ad[roi==1]
                metricInROI_mad_ad[r] = stats.median_absolute_deviation(1000*tmp_ad)
                metricInROI_median_ad[r] = np.median(1000*tmp_ad)
                metricInROI_mean_ad[r] = np.mean(1000*tmp_ad)
                metricInROI_std_ad[r] = np.std(1000*tmp_ad)
                metricInROI_weightedMean_ad[r] = np.sum(1000*tmp_ad * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_ad[r] = np.sqrt(np.mean((1000*tmp_ad - metricInROI_weightedMean_ad[r])**2))

            if type(rd)==np.ndarray:
                tmp_rd = rd[roi==1]
                metricInROI_mad_rd[r] = stats.median_absolute_deviation(1000*tmp_rd)
                metricInROI_median_rd[r] = np.median(1000*tmp_rd)
                metricInROI_mean_rd[r] = np.mean(1000*tmp_rd)
                metricInROI_std_rd[r] = np.std(1000*tmp_rd)
                metricInROI_weightedMean_rd[r] = np.sum(1000*tmp_rd * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_rd[r] = np.sqrt(np.mean((1000*tmp_rd - metricInROI_weightedMean_rd[r])**2))

            if type(rk)==np.ndarray:
                tmp_rk = rk[roi==1]
                metricInROI_mad_rk[r] = stats.median_absolute_deviation(tmp_rk)
                metricInROI_median_rk[r] = np.median(tmp_rk)
                metricInROI_mean_rk[r] = np.mean(tmp_rk)
                metricInROI_std_rk[r] = np.std(tmp_rk)
                metricInROI_weightedMean_rk[r] = np.sum(tmp_rk * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_rk[r] = np.sqrt(np.mean((tmp_rk - metricInROI_weightedMean_rk[r])**2))

            if type(ak)==np.ndarray:
                tmp_ak = ak[roi==1]
                metricInROI_mad_ak[r] = stats.median_absolute_deviation(tmp_ak)
                metricInROI_median_ak[r] = np.median(tmp_ak)
                metricInROI_mean_ak[r] = np.mean(tmp_ak)
                metricInROI_std_ak[r] = np.std(tmp_ak)
                metricInROI_weightedMean_ak[r] = np.sum(tmp_ak * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_ak[r] = np.sqrt(np.mean((tmp_ak - metricInROI_weightedMean_ak[r])**2))

            if type(mo)==np.ndarray:
                tmp_mo = mo[roi==1]
                metricInROI_mad_mo[r] = stats.median_absolute_deviation(tmp_mo)
                metricInROI_median_mo[r] = np.median(tmp_mo)
                metricInROI_mean_mo[r] = np.mean(tmp_mo)
                metricInROI_std_mo[r] = np.std(tmp_mo)
                metricInROI_weightedMean_mo[r] = np.sum(tmp_mo * probabilities) / np.sum(probabilities)
                metricInROI_weightedStd_mo[r] = np.sqrt(np.mean((tmp_mo - metricInROI_weightedMean_mo[r])**2))

        # save metrics in csv file
        with open(self.out_csv_name, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(np.hstack(['metric', np.arange(numberOfROIs)]))
            filewriter.writerow(np.hstack(['ROI_Volumes_mm3', ROI_volume_mm3]))
            filewriter.writerow(np.hstack(['ROI_Mad_FA', metricInROI_mad_fa]))
            filewriter.writerow(np.hstack(['ROI_Median_FA', metricInROI_median_fa]))
            filewriter.writerow(np.hstack(['ROI_Mean_FA', metricInROI_mean_fa]))
            filewriter.writerow(np.hstack(['ROI_Std_FA', metricInROI_std_fa]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_FA', metricInROI_weightedMean_fa]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_FA', metricInROI_weightedStd_fa]))
            filewriter.writerow(np.hstack(['ROI_Mad_MD', metricInROI_mad_md]))
            filewriter.writerow(np.hstack(['ROI_Median_MD', metricInROI_median_md]))
            filewriter.writerow(np.hstack(['ROI_Mean_MD', metricInROI_mean_md]))
            filewriter.writerow(np.hstack(['ROI_Std_MD', metricInROI_std_md]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_MD', metricInROI_weightedMean_md]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_MD', metricInROI_weightedStd_md]))
            filewriter.writerow(np.hstack(['ROI_Mad_MK', metricInROI_mad_mk]))
            filewriter.writerow(np.hstack(['ROI_Median_MK', metricInROI_median_mk]))
            filewriter.writerow(np.hstack(['ROI_Mean_MK', metricInROI_mean_mk]))
            filewriter.writerow(np.hstack(['ROI_Std_MK', metricInROI_std_mk]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_MK', metricInROI_weightedMean_mk]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_MK', metricInROI_weightedStd_mk]))
            filewriter.writerow(np.hstack(['ROI_Mad_RD', metricInROI_mad_rd]))
            filewriter.writerow(np.hstack(['ROI_Median_RD', metricInROI_median_rd]))
            filewriter.writerow(np.hstack(['ROI_Mean_RD', metricInROI_mean_rd]))
            filewriter.writerow(np.hstack(['ROI_Std_RD', metricInROI_std_rd]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_RD', metricInROI_weightedMean_rd]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_RD', metricInROI_weightedStd_rd]))
            filewriter.writerow(np.hstack(['ROI_Mad_AD', metricInROI_mad_ad]))
            filewriter.writerow(np.hstack(['ROI_Median_AD', metricInROI_median_ad]))
            filewriter.writerow(np.hstack(['ROI_Mean_AD', metricInROI_mean_ad]))
            filewriter.writerow(np.hstack(['ROI_Std_AD', metricInROI_std_ad]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_AD', metricInROI_weightedMean_ad]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_AD', metricInROI_weightedStd_ad]))
            filewriter.writerow(np.hstack(['ROI_Mad_RK', metricInROI_mad_rk]))
            filewriter.writerow(np.hstack(['ROI_Median_RK', metricInROI_median_rk]))
            filewriter.writerow(np.hstack(['ROI_Mean_RK', metricInROI_mean_rk]))
            filewriter.writerow(np.hstack(['ROI_Std_RK', metricInROI_std_rk]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_RK', metricInROI_weightedMean_rk]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_RK', metricInROI_weightedStd_rk]))
            filewriter.writerow(np.hstack(['ROI_Mad_AK', metricInROI_mad_ak]))
            filewriter.writerow(np.hstack(['ROI_Median_AK', metricInROI_median_ak]))
            filewriter.writerow(np.hstack(['ROI_Mean_AK', metricInROI_mean_ak]))
            filewriter.writerow(np.hstack(['ROI_Std_AK', metricInROI_std_ak]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_AK', metricInROI_weightedMean_ak]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_AK', metricInROI_weightedStd_ak]))
            filewriter.writerow(np.hstack(['ROI_Mad_MO', metricInROI_mad_mo]))
            filewriter.writerow(np.hstack(['ROI_Median_MO', metricInROI_median_mo]))
            filewriter.writerow(np.hstack(['ROI_Mean_MO', metricInROI_mean_mo]))
            filewriter.writerow(np.hstack(['ROI_Std_MO', metricInROI_std_mo]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Mean_MO', metricInROI_weightedMean_mo]))
            filewriter.writerow(np.hstack(['ROI_Weighted_Std_MO', metricInROI_weightedStd_mo]))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_csv_report'] = self.out_csv_name

        return outputs

    def _gen_filename(self, name):
        if name == 'out_csv_report':
            return self._gen_outfilename()
        return None
#-----------------------------------------------------------------------------------------------------#
# CREATE FODs 
#-----------------------------------------------------------------------------------------------------#
class CreateFodsInputSpec(BaseInterfaceInputSpec): 
    in_file = File(exists=True, desc='DWI volume to be parcellated', mandatory=True)
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(exists=True, desc='brain mask to calculate tracts in', mandatory=True)

class CreateFodsOutputSpec(TraitedSpec): 
    out_peaks = File(desc='peaks')

class CreateFods(BaseInterface):
    input_spec = CreateFodsInputSpec
    output_spec = CreateFodsOutputSpec

    def _run_interface(self, runtime):
        # define output folders
        _, base, _ = split_filename(self.inputs.in_file)
        output_dir = os.path.abspath('')

        # define names for output files 
        self.peaks = os.path.join(output_dir, 'peaks.nii.gz')

        # run the method to extract the peaks 
        from tractseg.libs import preprocessing
        preprocessing.create_fods(self.inputs.in_file, output_dir, self.inputs.in_bvals, 
                            self.inputs.in_bvecs, self.inputs.in_mask, 'csd', nr_cpus=-1) 
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_peaks'] = self.peaks
        return outputs

    def _gen_filename(self, name): 
        if name == 'out_peaks':
            return self._gen_outfilename()
        return None

#-----------------------------------------------------------------------------------------------------#
# FLIP PEAKS 
#-----------------------------------------------------------------------------------------------------#
class FlipPeaksInputSpec(BaseInterfaceInputSpec):
    in_peaks = File(exists=True, desc='peaks to be flipped', mandatory=True)
    #in_axis = traits.Str(decs='which axis that should be flipped')

class FlipPeaksOutputSpec(TraitedSpec): 
    out_flipped_peaks = File(desc='flipped_peaks')

class FlipPeaks(BaseInterface):
    input_spec = FlipPeaksInputSpec
    output_spec = FlipPeaksOutputSpec

    def _run_interface(self, runtime):
       # if isdefined(self.inputs.in_axis):
        #    axisFlip = "-a " + self.inputs.in_axis
        #else: 
        axisFlip = "-a x"

        # define output folder
        output_dir = os.path.abspath('')

        # define names for output files 
        self.flipped_peaks = os.path.join(output_dir, 'flipped_peaks.nii.gz')

        # initialise and run ROBEX 
        _flippeaks = FLIPPEAKS(input_peaks=self.inputs.in_peaks, input_axis = axisFlip, output_file=self.flipped_peaks)
        _flippeaks.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_flipped_peaks'] = self.flipped_peaks
        return outputs

    def _gen_filename(self, name):
        if name == 'out_flipped_peaks':
            return self._gen_outfilename()
        return None

class FLIPPEAKSInputSpec(CommandLineInputSpec):
    input_peaks = traits.Str(exists=True, desc='peaks to be flipped',
                            mandatory=True, argstr="-i %s", position=0)
    input_axis = traits.Str(desc='which axis to be flipped', 
                            mandatory=True, argstr="%s", position=2)
    output_file = traits.Str(desc='flipped peaks', 
                            mandatory=True, argstr="-o %s", position=1)

class FLIPPEAKS(CommandLine):
    input_spec = FLIPPEAKSInputSpec
    _cmd = 'python flip_peaks '
#
#-----------------------------------------------------------------------------------------------------#
# PEAKS TO SEGMENTATION
#-----------------------------------------------------------------------------------------------------#

class PeaksTractSegInputSpec(BaseInterfaceInputSpec):
    in_peaks = File(exists=True, desc='peaks as input to TractSeg', mandatory=True)
    in_file = File(exists=True, desc='DWI image', mandatory=True)

class PeaksTractSegOutputSpec(TraitedSpec):
    out_probability_atlas = File(desc='tract probability atlas 4D')
    out_binary_atlas = File(desc='tract segmentation  atlas 4D')
    out_mosaic_tracts = File(desc='mosaic file with 2d-projections of all tracts')

class PeaksTractSeg(BaseInterface):
    input_spec = PeaksTractSegInputSpec
    output_spec = PeaksTractSegOutputSpec

    def _run_interface(self, runtime):

        # define output folders
        output_dir = os.path.abspath('') 
        
        # define names for output files
        self.probability_atlas = os.path.join(output_dir, 'bundle_segmentations.nii.gz')
        self.binary_atlas = os.path.join(output_dir, 'bundle_segmentations_binary.nii.gz')
        self.mosaic_tracts = os.path.join(output_dir, 'mosaic_tracts.png')

        # initialise and run ROBEX
        _tractseg = PEAKTRACTSEG(input_peaks=self.inputs.in_peaks,
            output_dir=output_dir)
        _tractseg.run()
        
        # save probabilty atlas 
        atlas_proxy = nib.load(self.probability_atlas)
        probability_atlas = np.squeeze(atlas_proxy.get_data())
        
        # create binary atlas by thresholding the probability atlas, 
        binary_temp = np.where(probability_atlas >0.5, 1, 0)
        
        # postprocessing is done in the same way as it is done in TractSeg 
        from tractseg.libs import img_utils
        from tractseg.data import dataset_specific_utils
        
        binary_temp = img_utils.bundle_specific_postprocessing(binary_temp, 
                dataset_specific_utils.get_bundle_names("All")[1:])
        binary_atlas = img_utils.postprocess_segmentations(binary_temp, 
                dataset_specific_utils.get_bundle_names("All")[1:], blob_thr=50, hole_closing=None)
        
        # binary segmentation is saved
        binary_atlasFile = nib.Nifti1Image(binary_atlas, atlas_proxy.affine, atlas_proxy.get_header())
        nib.save(binary_atlasFile, self.binary_atlas)
        
        # create mosaic file with 2D-projections of all tracts in coronal view
        from tractseg.data.dataset_specific_utils import get_bundle_names
        import math 
        import matplotlib.pyplot as plt
        
        background_img = nib.load(self.inputs.in_file).get_fdata()
        background_img = background_img[:,:,:,0] 
        plt.ioff() # Turn the interactive mode off
        plt.style.use('dark_background')
        
        bundles = get_bundle_names("All")[1:]
        cols = 10
        rows = math.ceil(len(bundles)/cols)
        
        def plot_single_tract(bg, data, orientation, bundle):
        
            if orientation == "coronal":
                data = data.transpose(2, 0, 1)
                data = data[::-1, :, :]
                bg = bg.transpose(2, 0, 1)[::-1, :, :]
            elif orientation == "sagittal":
                data = data.transpose(2, 1, 0)
                data = data[::-1, :, :]
                bg = bg.transpose(2, 1, 0)[::-1, :, :]
            else:  # axial
                pass
        
            # Determines the center slice of the tract
            mask_voxel_coords = np.where(data != 0)
            if len(mask_voxel_coords) > 2 and len(mask_voxel_coords[2]) > 0:
                minidx = int(np.min(mask_voxel_coords[2]))
                maxidx = int(np.max(mask_voxel_coords[2])) + 1
                mean_slice = int(np.mean([minidx, maxidx]))
            else:
                mean_slice = int(bg.shape[2] / 2)
            bg = bg[:, :, mean_slice]
        
            # project 3D to 2D image
            data = data.max(axis=2)
        
            font = {'color':'white', 'size':4, 'verticalalignment' : 'baseline' }
            plt.imshow(bg, cmap="gray")
            data = np.ma.masked_where(data < 0.00001, data)
            plt.text(0,0, bundle, fontdict=font)
            plt.imshow(data, cmap="autumn")
        
        for j, bundle in enumerate(bundles): 
            orientation = "coronal" # could change this for the different tracts
            bundle_idx = get_bundle_names("All")[1:].index(bundle)
            mask_data = binary_atlas[:, :, :, bundle_idx]
            mask_data = np.copy(mask_data)  # copy data otherwise will also threshold data outside of plot function
        
            plt.subplot(rows, cols, j+1)
        
            plt.axis("off")
            plt.tight_layout()
            plot_single_tract(background_img, mask_data, orientation, bundle) 
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("mosaic_tracts.png", bbox_inches='tight', dpi=300)
        
        return runtime
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_probability_atlas'] = self.probability_atlas
        outputs['out_binary_atlas'] = self.binary_atlas
        outputs['out_mosaic_tracts'] = self.mosaic_tracts
        return outputs

    def _gen_filename(self, name):
        if name == 'out_probability_atlas':
            return self._gen_outfilename()
        return None

class PEAKTRACTSEGinputSpec(CommandLineInputSpec):
    input_peaks = traits.Str(exists=True, desc='input peaks', mandatory=True, argstr="-i %s", position=0)
    output_dir = traits.Str(desc='output directory', mandatory=True, argstr="-o %s", position=4)

class PEAKTRACTSEG(CommandLine):
    input_spec = PEAKTRACTSEGinputSpec
    _cmd = 'python /home/fuji/Software/tractseg/TractSeg/bin/TractSeg --get_probabilities --single_output_file'

#-----------------------------------------------------------------------------------------------------#
# Smooth DTI
#-----------------------------------------------------------------------------------------------------#
from dipy.io import read_bvals_bvecs
from dipy.data import gradient_table
from scipy.ndimage.filters import gaussian_filter

class DTIsmoothInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='DTI volume to be smoothed', mandatory=True)
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(exists=True, desc='brain mask', mandatory=True)
    in_smooth = traits.Float(exists=True, desc='smoothing factor', mandatory=False)

class DTIsmoothOutputSpec(TraitedSpec):
    out_dti = File(exists=True, desc='smooth DTI', mandatory=True)

class DTIsmooth(BaseInterface):
    input_spec = DTIsmoothInputSpec
    output_spec = DTIsmoothOutputSpec

    def _run_interface(self, runtime):
        # define output folder
        output_dir = os.path.abspath('')

        # define names for output files 
        self.out_dti = os.path.join(output_dir, 'smooth_DTI.nii.gz')
    
        # read b-values and b-vectors
        bvals, bvecs = read_bvals_bvecs(self.inputs.in_bvals, self.inputs.in_bvecs)
        gtab = gradient_table(bvals, bvecs)
        
        # load file and mask
        dti_proxy = nib.load(self.inputs.in_file)
        dti_data = dti_proxy.get_data()
        dti_mask_proxy = nib.load(self.inputs.in_mask)
        dti_mask = dti_mask_proxy.get_data()
        
        # smooth data
        if self.inputs.in_smooth:
            gauss_std = self.inputs.in_smooth
        else:
            gauss_std = 0.75
        data_smooth = np.zeros(dti_data.shape)
        for v in range(dti_data.shape[-1]):
            data_smooth[..., v] = gaussian_filter(dti_data[..., v], sigma=gauss_std)

        dti_new = nib.Nifti1Image(data_smooth, dti_proxy.affine, dti_proxy.get_header())
        nib.save(dti_new, self.out_dti)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_dti'] = self.out_dti
        return outputs

    def _gen_filename(self, name):
        if name == 'out_dti':
            return self._gen_outfilename()
        return None

#-----------------------------------------------------------------------------------------------------#
# DKI FIT smooth
#-----------------------------------------------------------------------------------------------------#
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.io import read_bvals_bvecs
from dipy.data import gradient_table
from scipy.ndimage.filters import gaussian_filter


class DKIfitSmoothInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='DKI volume to be analysed', mandatory=True)
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(exists=True, desc='brain mask', mandatory=True)
    in_smooth = traits.Float(exists=True, desc='smoothing factor', mandatory=False)

class DKIfitSmoothOutputSpec(TraitedSpec):
    out_fa = File(exists=True, desc='FA map', mandatory=True)
    out_md = File(exists=True, desc='MD map', mandatory=True)
    out_mk = File(exists=True, desc='MK map', mandatory=True)
    out_rd = File(exists=True, desc='RD map', mandatory=True)
    out_ad = File(exists=True, desc='AD map', mandatory=True) 
    out_ak = File(exists=True, desc='AK map', mandatory=True) 
    out_rk = File(exists=True, desc='RK map', mandatory=True) 
    out_sm = File(exists=True, desc='Smooth map', mandatory=True) 

class DKIfitSmooth(BaseInterface):
    input_spec = DKIfitSmoothInputSpec
    output_spec = DKIfitSmoothOutputSpec

    def _run_interface(self, runtime):
        # define output folder
        output_dir = os.path.abspath('')

        # define names for output files 
        self.fa = os.path.join(output_dir, 'dkifit_FA.nii.gz')
        self.md = os.path.join(output_dir, 'dkifit_MD.nii.gz')
        self.mk = os.path.join(output_dir, 'dkifit_MK.nii.gz')
        self.ad = os.path.join(output_dir, 'dkifit_AD.nii.gz')
        self.rd = os.path.join(output_dir, 'dkifit_RD.nii.gz')
        self.rk = os.path.join(output_dir, 'dkifit_RK.nii.gz')
        self.ak = os.path.join(output_dir, 'dkifit_AK.nii.gz')
        self.sm = os.path.join(output_dir, 'smoothDWI.nii.gz')
    
        # read b-values and b-vectors
        bvals, bvecs = read_bvals_bvecs(self.inputs.in_bvals, self.inputs.in_bvecs)
        gtab = gradient_table(bvals, bvecs)
        
        # load file and mask
        dki_proxy = nib.load(self.inputs.in_file)
        dki_data = dki_proxy.get_data()
        dki_mask_proxy = nib.load(self.inputs.in_mask)
        dki_mask = dki_mask_proxy.get_data()
        
        # smooth data
        if self.inputs.in_smooth:
            gauss_std = self.inputs.in_smooth
        else:
            gauss_std = 0.75
        data_smooth = np.zeros(dki_data.shape)
        for v in range(dki_data.shape[-1]):
            data_smooth[..., v] = gaussian_filter(dki_data[..., v], sigma=gauss_std)

        dkimodel = DiffusionKurtosisModel(gtab, fit_method="WLS")
        dkifit = dkimodel.fit(data_smooth, dki_mask)
        
        dki_fa = nib.Nifti1Image(dkifit.fa, dki_proxy.affine, dki_proxy.get_header())
        dki_md = nib.Nifti1Image(dkifit.md, dki_proxy.affine, dki_proxy.get_header())
        dki_ad = nib.Nifti1Image(dkifit.ad, dki_proxy.affine, dki_proxy.get_header())
        dki_rd = nib.Nifti1Image(dkifit.rd, dki_proxy.affine, dki_proxy.get_header())
        dki_mk = nib.Nifti1Image(dkifit.mk(0,3), dki_proxy.affine, dki_proxy.get_header())
        dki_rk = nib.Nifti1Image(dkifit.rk(0,3), dki_proxy.affine, dki_proxy.get_header())
        dki_ak = nib.Nifti1Image(dkifit.ak(0,3), dki_proxy.affine, dki_proxy.get_header())
        dki_sm = nib.Nifti1Image(data_smooth, dki_proxy.affine, dki_proxy.get_header())
        nib.save(dki_fa, self.fa)
        nib.save(dki_md, self.md)
        nib.save(dki_mk, self.mk)
        nib.save(dki_ad, self.ad)
        nib.save(dki_rd, self.rd)
        nib.save(dki_rk, self.rk)
        nib.save(dki_ak, self.ak)
        nib.save(dki_sm, self.sm)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_fa'] = self.fa
        outputs['out_md'] = self.md
        outputs['out_mk'] = self.mk
        outputs['out_ad'] = self.ad
        outputs['out_rd'] = self.rd
        outputs['out_ak'] = self.ak
        outputs['out_rk'] = self.rk
        outputs['out_sm'] = self.sm
        return outputs

    def _gen_filename(self, name):
        if name == 'out_fa':
            return self._gen_outfilename()
        return None

#-----------------------------------------------------------------------------------------------------#
# DKI FIT
#-----------------------------------------------------------------------------------------------------#
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.io import read_bvals_bvecs
from dipy.data import gradient_table
from scipy.ndimage.filters import gaussian_filter


class DKIfitInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='DKI volume to be analysed', mandatory=True)
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(exists=True, desc='brain mask', mandatory=True)
    in_smooth = traits.Float(exists=True, desc='smoothing factor', mandatory=False)

class DKIfitOutputSpec(TraitedSpec):
    out_fa = File(exists=True, desc='FA map', mandatory=True)
    out_md = File(exists=True, desc='MD map', mandatory=True)
    out_mk = File(exists=True, desc='MK map', mandatory=True)
    out_rd = File(exists=True, desc='RD map', mandatory=True)
    out_ad = File(exists=True, desc='AD map', mandatory=True) 
    out_rk = File(exists=True, desc='RK map', mandatory=True) 
    out_ak = File(exists=True, desc='AK map', mandatory=True) 

class DKIfit(BaseInterface):
    input_spec = DKIfitInputSpec
    output_spec = DKIfitOutputSpec

    def _run_interface(self, runtime):
        # define output folder
        output_dir = os.path.abspath('')

        # define names for output files 
        self.fa = os.path.join(output_dir, 'dkifit_FA.nii.gz')
        self.md = os.path.join(output_dir, 'dkifit_MD.nii.gz')
        self.mk = os.path.join(output_dir, 'dkifit_MK.nii.gz')
        self.ad = os.path.join(output_dir, 'dkifit_AD.nii.gz')
        self.rd = os.path.join(output_dir, 'dkifit_RD.nii.gz')
        self.ak = os.path.join(output_dir, 'dkifit_AK.nii.gz')
        self.rk = os.path.join(output_dir, 'dkifit_RK.nii.gz')
    
        # read b-values and b-vectors
        bvals, bvecs = read_bvals_bvecs(self.inputs.in_bvals, self.inputs.in_bvecs)
        gtab = gradient_table(bvals, bvecs)
        
        # load file and mask
        dki_proxy = nib.load(self.inputs.in_file)
        dki_data = dki_proxy.get_data()
        dki_mask_proxy = nib.load(self.inputs.in_mask)
        dki_mask = dki_mask_proxy.get_data()
        
       
        dkimodel = DiffusionKurtosisModel(gtab, fit_method="WLS")
        dkifit = dkimodel.fit(dki_data, dki_mask)
        
        dki_fa = nib.Nifti1Image(dkifit.fa, dki_proxy.affine, dki_proxy.get_header())
        dki_md = nib.Nifti1Image(dkifit.md, dki_proxy.affine, dki_proxy.get_header())
        dki_ad = nib.Nifti1Image(dkifit.ad, dki_proxy.affine, dki_proxy.get_header())
        dki_rd = nib.Nifti1Image(dkifit.rd, dki_proxy.affine, dki_proxy.get_header())
        dki_mk = nib.Nifti1Image(dkifit.mk(0,3), dki_proxy.affine, dki_proxy.get_header())
        dki_ak = nib.Nifti1Image(dkifit.ak(0,3), dki_proxy.affine, dki_proxy.get_header())
        dki_rk = nib.Nifti1Image(dkifit.rk(0,3), dki_proxy.affine, dki_proxy.get_header())
        nib.save(dki_fa, self.fa)
        nib.save(dki_md, self.md)
        nib.save(dki_mk, self.mk)
        nib.save(dki_ad, self.ad)
        nib.save(dki_rd, self.rd)
        nib.save(dki_ak, self.ak)
        nib.save(dki_rk, self.rk)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_fa'] = self.fa
        outputs['out_md'] = self.md
        outputs['out_mk'] = self.mk
        outputs['out_rk'] = self.rk
        outputs['out_ak'] = self.ak
        outputs['out_ad'] = self.ad
        outputs['out_rd'] = self.rd
        return outputs

    def _gen_filename(self, name):
        if name == 'out_fa':
            return self._gen_outfilename()
        return None

#-----------------------------------------------------------------------------------------------------#
# FREE WATER ELIMINATION
#-----------------------------------------------------------------------------------------------------#
class FreeWaterEliminationInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='dwi volume for dti fitting', mandatory=True)
    in_bval = traits.Str(exists=True, desc='bvals', mandatory=True)
    in_bvec = traits.Str(exists=True, desc='bvecs', mandatory=True)
    in_mask = File(exists=True, desc='brain mask', mandatory=True)
    threshold = traits.Float(desc='threshold for artifact removal')

class FreeWaterEliminationOutputSpec(TraitedSpec):
    out_FA = File(exists=True, desc='fractional anisotropy map with free water elimination')
    out_MD = File(exists=True, desc='mean diffusivity map with free water elimination')
    out_FW = File(exists=True, desc='free water contamination map')
    out_eVecs = File(exists=True, desc='DTI vectors with free water elimination')
    out_eVals = File(exists=True, desc='DTI eigen values with free water elimination')
    out_FA_thresholded = File(exists=True, desc='fractional anisotropy map with free water elimination thresholded')
    out_fwmask = File(exists=True, desc='threshold freewater map mask')
    out_mask = File(exists=True, desc='threshold freewater map mask')

class FreeWaterElimination(BaseInterface):
    input_spec = FreeWaterEliminationInputSpec
    output_spec = FreeWaterEliminationOutputSpec

    def _run_interface(self, runtime):

        # load image and get predefined slices
        imgFile = nib.load(self.inputs.in_file)
        img = imgFile.get_data()
        #_, base, _ = split_filename(self.inputs.in_file)

        mask = nib.load(self.inputs.in_mask).get_data()

        if self.inputs.in_bvec[-6:] == '.bvecs':
            in_bvec = self.inputs.in_bvec[:-1]
        elif self.inputs.in_bvec[-19:] == '.eddy_rotated_bvecs':
            dst = self.inputs.in_bvec[-19] + '.bvec'
            shutil.copy(self.inputs.in_bvec, dst)
            in_bvec = dst
        else:
            in_bvec = self.inputs.in_bvec

        if self.inputs.in_bval[-6:] == '.bvals':
            in_bval = self.inputs.in_bval[:-1]
        else:
            in_bval = self.inputs.in_bval

        if isdefined(self.inputs.threshold):
            threshold = self.inputs.threshold
        else:
            threshold = 0.7

        # load bvecs
        gtab = gradient_table(in_bval, in_bvec)

        # fit model
        fwdtimodel = fwdti.FreeWaterTensorModel(gtab)
        fwdtifit = fwdtimodel.fit(img, mask=mask)

        FA = fwdtifit.fa
        MD = fwdtifit.md
        FW = fwdtifit.f # free water contamination map
        evals = fwdtifit.evals
        evecs =  fwdtifit.evecs

        # threhsold FA map to remove artefacts of regions associated to voxels with
        # high water volume fraction (i.e. voxels containing basically CSF).
        thFA = copy.copy(FA)
        thFA[FW > threshold] = 0
        fwMask = np.where(FW>threshold,1, 0)
        nfwMask = np.abs(fwMask-1)

          # save images to folder
        faFile = nib.Nifti1Image(FA, imgFile.affine, imgFile.get_header())
        nib.save(faFile, 'fwFA.nii')
        self.FA = os.path.abspath('fwFA.nii')

        mdFile = nib.Nifti1Image(MD, imgFile.affine, imgFile.get_header())
        nib.save(mdFile, 'fwMD.nii')
        self.MD = os.path.abspath('fwMD.nii')

        fwFile = nib.Nifti1Image(FW, imgFile.affine, imgFile.get_header())
        nib.save(fwFile,'FW.nii')
        self.FW = os.path.abspath('FW.nii')

        evalsFile = nib.Nifti1Image(evals, imgFile.affine, imgFile.get_header())
        nib.save(evalsFile,'eVals.nii')
        self.eVals = os.path.abspath('eVals.nii')

        evecsFile = nib.Nifti1Image(evecs, imgFile.affine, imgFile.get_header())
        nib.save(evecsFile,'eVecs.nii')
        self.eVecs = os.path.abspath('eVecs.nii')

        thFaFile = nib.Nifti1Image(thFA, imgFile.affine, imgFile.get_header())
        nib.save(thFaFile, 'fwFA_thresholded_'+ str(threshold) +'.nii')
        self.thFA = os.path.abspath('fwFA_thresholded_'+ str(threshold) +'.nii')

        fwMaskFile = nib.Nifti1Image(fwMask, imgFile.affine, imgFile.get_header())
        nib.save(fwMaskFile, 'freeWater_mask_thr_'+ str(threshold) +'.nii')
        self.fwMask = os.path.abspath('freeWater_mask_thr_'+ str(threshold) +'.nii')

        nfwMaskFile = nib.Nifti1Image(nfwMask, imgFile.affine, imgFile.get_header())
        nib.save(nfwMaskFile, 'negativeFreeWater_mask_thr_'+ str(threshold) +'.nii')
        self.nfwMask = os.path.abspath('negativeFreeWater_mask_thr_'+ str(threshold) +'.nii')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_FA'] = self.FA
        outputs['out_MD'] = self.MD
        outputs['out_FW'] = self.FW
        outputs['out_eVals'] = self.eVals
        outputs['out_eVecs'] = self.eVecs
        outputs['out_FA_thresholded'] = self.thFA
        outputs['out_fwmask'] = self.fwMask
        outputs['out_mask'] = self.nfwMask

        return outputs

    def _gen_filename(self, name):
        if name == 'out_FA':
            return self._gen_outfilename()

        return None


#-----------------------------------------------------------------------------------------------------#
# Intravoxel incoherent motion
#-----------------------------------------------------------------------------------------------------#
class IntraVoxelIncoherentMotionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='dwi volume for dti fitting', mandatory=True)
    in_bval = traits.Str(exists=True, desc='bvals', mandatory=True)
    in_bvec = traits.Str(exists=True, desc='bvecs', mandatory=True)

class IntraVoxelIncoherentMotionOutputSpec(TraitedSpec):
    out_predicted_S0 = File(exists=True, desc='')
    out_perfusion_fraction = File(exists=True, desc='')
    out_perfusion_coeff = File(exists=True, desc='')
    out_diffusion_coeff = File(exists=True, desc='')

class IntraVoxelIncoherentMotion(BaseInterface):
    input_spec = IntraVoxelIncoherentMotionInputSpec
    output_spec = IntraVoxelIncoherentMotionOutputSpec

    def _run_interface(self, runtime):

        # load image and get predefined slices
        imgFile = nib.load(self.inputs.in_file)
        img = imgFile.get_data()

        if self.inputs.in_bvec[-1] == 's':
            in_bvec = self.inputs.in_bvec[:-1]
        else:
            in_bvec = self.inputs.in_bvec

        if self.inputs.in_bval[-1] == 's':
            in_bval = self.inputs.in_bval[:-1]
        else:
            in_bval = self.inputs.in_bval

        # load bvecs
        gtab = gradient_table(in_bval, in_bvec)

        # fit model
        ivimmodel = IvimModel(gtab)
        ivimfit = ivimmodel.fit(img)


        S0 = ivimfit.S0_predicted
        PF = ivimfit.perfusion_fraction
        perfCoeff = ivimfit.D_star
        diffCoeff = ivimfit.D

        # save images to folder
        s0File = nib.Nifti1Image(S0, imgFile.affine, imgFile.get_header())
        nib.save(s0File, 'predicted_S0.nii')
        self.S0 = os.path.abspath('predicted_S0.nii')

        pfFile = nib.Nifti1Image(PF, imgFile.affine, imgFile.get_header())
        nib.save(pfFile, 'perfusion_fraction.nii')
        self.PF = os.path.abspath('perfusion_fraction.nii')

        perfCoeffFile = nib.Nifti1Image(perfCoeff, imgFile.affine, imgFile.get_header())
        nib.save(perfCoeffFile, 'perfusion_coeff.nii')
        self.perfCoeff = os.path.abspath('perfusion_coeff.nii')


        diffCoeffFile = nib.Nifti1Image(diffCoeff, imgFile.affine, imgFile.get_header())
        nib.save(diffCoeffFile, 'diffusion_coeff.nii')
        self.diffCoeff = os.path.abspath('diffusion_coeff.nii')


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_predicted_S0'] = self.S0
        outputs['out_perfusion_fraction'] = self.PF
        outputs['out_perfusion_coeff'] = self.perfCoeff
        outputs['out_diffusion_coeff'] = self.diffCoeff
        return outputs

    def _gen_filename(self, name):
        if name == 'out_predicted_S0':
            return self._gen_outfilename()

        return None


#-----------------------------------------------------------------------------------------------------#
# DTI-TK Preparation
#-----------------------------------------------------------------------------------------------------#
class PrepareDTITKInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='DTI volume to compute tensor image for', mandatory=True)

class PrepareDTITKOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='dtitk file')
    out_nonSPD = File(exists=True, desc='non symmetric and positive-definite matrice voxels')
    out_norm = File(exists=True, desc='tensor norm image')
    out_nonOutliers = File(exists=True, desc='mask excluding outlier voxels')
    #DOC: http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.BeforeReg

class PrepareDTITK(BaseInterface):
    input_spec = PrepareDTITKInputSpec
    output_spec = PrepareDTITKOutputSpec

    def _run_interface(self, runtime):

        # define output folders
        path, base, _ = split_filename(self.inputs.in_file)
        output_dir = os.path.abspath('')
        splitBase = base.split('_')[0]
        dtitkBase = splitBase + '_dtitk'
        tensorFile = os.path.join(path, splitBase)

        # define names for output files
        self.out_file = os.path.join(output_dir, dtitkBase + '.nii.gz')
        self.out_nonSPD = os.path.join(output_dir, dtitkBase + '_nonSPD.nii.gz')
        self.out_norm = os.path.join(output_dir, dtitkBase + '_norm.nii.gz')
        self.out_nonOutliers = os.path.join(output_dir, dtitkBase + '_norm_non_outliers.nii.gz')

        # initialise and run commands
        _dtiprep = dtitkPreparation(input_file=tensorFile)
        _dtiprep.run()
        _dtimove = dtitkMove(input_files=tensorFile +'_dtitk*', output_folder=output_dir)
        _dtimove.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        outputs['out_nonSPD'] = self.out_nonSPD
        outputs['out_norm'] = self.out_norm
        outputs['out_nonOutliers'] = self.out_nonOutliers

        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None

class dtitkPreparationInputSpec(CommandLineInputSpec):
    input_file = traits.Str(exists=True, desc='path and tensor base for dtifit FSL files',
                            mandatory=True, argstr="%s", position=0)

class dtitkPreparation(CommandLine):
    input_spec = dtitkPreparationInputSpec
    _cmd = 'fsl_to_dtitk'

class dtitkMoveInputSpec(CommandLineInputSpec):
    input_files = traits.Str(exists=True, desc='path and tensor base for dtifit FSL files',
                            mandatory=True, argstr="%s", position=0)
    output_folder = traits.Str(exists=True, desc='path to copy dtitk files to',
                            mandatory=True, argstr="%s", position=1)

class dtitkMove(CommandLine):
    input_spec = dtitkMoveInputSpec
    _cmd = 'mv'


#-----------------------------------------------------------------------------------------------------#
# DWI MASK: mask for FA>1 and L1<0
#-----------------------------------------------------------------------------------------------------#
class DWIMaskInputSpec(BaseInterfaceInputSpec):
    in_FA = File(exists=True, desc='FA volume', mandatory=True)
    in_L1 = File(exists=True, desc='L1 volume', mandatory=True)

class DWIMaskOutputSpec(TraitedSpec):
    out_artifcatMask = File(exists=True, desc='')
    out_mask = File(exists=True, desc='')
    out_numberOutlier = traits.Float(exists=True, desc='')


class DWIMask(BaseInterface):
    input_spec = DWIMaskInputSpec
    output_spec = DWIMaskOutputSpec

    def _run_interface(self, runtime):

        # load image and get predefined slices
        imgFile_FA = nib.load(self.inputs.in_FA)
        FA = imgFile_FA.get_data()
        _, base, _ = split_filename(self.inputs.in_FA)
        L1 = nib.load(self.inputs.in_L1).get_data()

        # find voxels with FA>1 (set to 1 in mask)
        faMask = np.zeros(FA.shape, dtype=int)
        ind = np.where(FA > 1)
        faMask[ind] = np.ones(len(ind[0]), dtype=int)

           # find voxels with L1<0 (set to 2 in mask)
        l1Mask = np.zeros(L1.shape, dtype=int)
        ind = np.where(L1 < 0)
        l1Mask[ind] = np.ones(len(ind[0]), dtype=int) * 2

           # combine masks: 1: FA>1, 2: L1<0, 3: FA>1 and L1<0
        artefactMask = faMask + l1Mask

        # count number of outlier voxels
        self.numberOutlier = np.sum(np.where(artefactMask > 0, 1, 0))

           # create mask that only includes voxels with FA<=1 and L1>0
        mask = np.where(artefactMask > 0, 0, 1) * np.where(FA > 0, 1, 0)

        # save image to folder
        artefactMaskFile = nib.Nifti1Image(artefactMask, imgFile_FA.affine, imgFile_FA.get_header())
        nib.save(artefactMaskFile, 'DTI_artefact_mask.nii.gz')
        self.artefactMask = os.path.abspath('DTI_artefact_mask.nii.gz')

        maskFile = nib.Nifti1Image(mask, imgFile_FA.affine, imgFile_FA.get_header())
        nib.save(maskFile, 'DTI_brain_mask.nii.gz')
        self.mask = os.path.abspath('DTI_brain_mask.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_artifcatMask'] = self.artefactMask
        outputs['out_mask'] = self.mask
        outputs['out_numberOutlier'] = self.numberOutlier
        return outputs

#-----------------------------------------------------------------------------------------------------#
# COMPOSITE TRANSFORMATIONS
#-----------------------------------------------------------------------------------------------------#
class CompositeTransformationsInputSpec(BaseInterfaceInputSpec):
    in_deform = File(exists=True, desc='non-linear deformation field', mandatory=True)
    in_affine = File(exists=True, desc='affine transformation file', mandatory=True)
    in_rigid = File(exists=True, desc='rigid transformation file', mandatory=True)

class CompositeTransformationsOutputSpec(TraitedSpec):
    out_composite = File(desc='output composite file')

class CompositeTransformations(BaseInterface):
    input_spec = CompositeTransformationsInputSpec
    output_spec = CompositeTransformationsOutputSpec

    def _run_interface(self, runtime):

        # define output folders
        _, base, _ = split_filename(self.inputs.in_deform)
        folder_output_dir = os.path.abspath('')
        output_dir = os.path.join(folder_output_dir) 
        
        # define names for output files
        self.composite = os.path.join(output_dir, 'composite_DWItoMNI.h5')

        _composite = Composite(input_rigid=self.inputs.in_rigid,
            input_affine=self.inputs.in_affine,
            input_deform=self.inputs.in_deform,
            output=self.composite)
        _composite.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_composite'] = self.composite

        return outputs

    def _gen_filename(self, name):
        if name == 'out_composite':
            return self._gen_outfilename()
        return None

class CompositeinputSpec(CommandLineInputSpec):
    input_rigid = traits.Str(exists=True, desc='input rigid transformation', mandatory=True, argstr="%s", position=1)
    input_affine = traits.Str(exists=True, desc='input affine transformation', mandatory=True, argstr="%s", position=2)
    input_deform = traits.Str(exists=True, desc='input deformation field', mandatory=True, argstr="%s", position=3)
    output = traits.Str(desc='output directory', mandatory=True, argstr="%s", position=0)

class Composite(CommandLine):
    input_spec = CompositeinputSpec
    _cmd = 'CompositeTransformUtil --assemble'

#-----------------------------------------------------------------------------------------------------#
# APPLY MEDIAN FILTER ON MK
#-----------------------------------------------------------------------------------------------------#
class ApplyMedianOnMKInputSpec(BaseInterfaceInputSpec):
    in_mk = traits.File(exists=True, desc='MK volume', mandatory=True)

class ApplyMedianOnMKOutputSpec(TraitedSpec):
    medfilt_mk = traits.File(exists=True, desc='MK map')

class ApplyMedianOnMK(BaseInterface):
    input_spec = ApplyMedianOnMKInputSpec
    output_spec = ApplyMedianOnMKOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.in_mk)
        output_dir = os.path.abspath('')
        self.medfilt_mk = os.path.join(output_dir, 'dkifit_MK_medfilt.nii.gz')
        mk = self.inputs.in_mk
        mk_load = nib.load(mk)
        mk_img = mk_load.get_data()
        mk_filtered = signal.medfilt(mk_img)
        ind = mk_img < (0.8*mk_filtered)
        mk_final = np.copy(mk_img)
        mk_final[ind] = mk_filtered[ind]
        mk_file = nib.Nifti1Image(mk_final, mk_load.affine, mk_load.get_header())
        nib.save(mk_file, self.medfilt_mk)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['medfilt_mk'] = self.medfilt_mk

        return outputs

#-----------------------------------------------------------------------------------------------------#
# APPLY MEDIAN FILTER ON AK
#-----------------------------------------------------------------------------------------------------#
class ApplyMedianOnAKInputSpec(BaseInterfaceInputSpec):
    in_ak = traits.File(exists=True, desc='AK volume', mandatory=True)

class ApplyMedianOnAKOutputSpec(TraitedSpec):
    medfilt_ak = traits.File(exists=True, desc='AK map')

class ApplyMedianOnAK(BaseInterface):
    input_spec = ApplyMedianOnAKInputSpec
    output_spec = ApplyMedianOnAKOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.in_ak)
        output_dir = os.path.abspath('')
        self.medfilt_ak = os.path.join(output_dir, 'dkifit_AK_medfilt.nii.gz')
        ak = self.inputs.in_ak
        ak_load = nib.load(ak)
        ak_img = ak_load.get_data()
        ak_filtered = signal.medfilt(ak_img)
        ind = ak_img < (0.8*ak_filtered)
        ak_final = np.copy(ak_img)
        ak_final[ind] = ak_filtered[ind]
        ak_file = nib.Nifti1Image(ak_final, ak_load.affine, ak_load.get_header())
        nib.save(ak_file, self.medfilt_ak)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['medfilt_ak'] = self.medfilt_ak

        return outputs

#-----------------------------------------------------------------------------------------------------#
# APPLY MEDIAN FILTER ON RK
#-----------------------------------------------------------------------------------------------------#
class ApplyMedianOnRKInputSpec(BaseInterfaceInputSpec):
    in_rk = traits.File(exists=True, desc='RK volume', mandatory=True)

class ApplyMedianOnRKOutputSpec(TraitedSpec):
    medfilt_rk = traits.File(exists=True, desc='RK map')

class ApplyMedianOnRK(BaseInterface):
    input_spec = ApplyMedianOnRKInputSpec
    output_spec = ApplyMedianOnRKOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.in_rk)
        output_dir = os.path.abspath('')
        self.medfilt_rk = os.path.join(output_dir, 'dkifit_RK_medfilt.nii.gz')
        rk = self.inputs.in_rk
        rk_load = nib.load(rk)
        rk_img = rk_load.get_data()
        rk_filtered = signal.medfilt(rk_img)
        ind = rk_img < (0.8*rk_filtered)
        rk_final = np.copy(rk_img)
        rk_final[ind] = rk_filtered[ind]
        rk_file = nib.Nifti1Image(rk_final, rk_load.affine, rk_load.get_header())
        nib.save(rk_file, self.medfilt_rk)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['medfilt_rk'] = self.medfilt_rk

        return outputs

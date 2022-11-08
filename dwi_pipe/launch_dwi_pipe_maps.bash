#! /bin/bash
# lines 3 to 13 define the initial values of the variables
scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
dirData=$1
IFS="/" read -r -a splitInit <<< "$dirData"
toStart=${#splitInit[@]}
DWIname=$2
dtiMaskName=$3
ccase=$4
smooth=$5
txt=$6
count=1
sessions=()
# lines 15 to 35 create the list of sessions/subjsects that are about to be processed by the pipeline
if [ $txt != "none" ]; then
  while IFS='' read -r line || [[ -n "$line" ]]; do
    session=$line
    T1w=$(find $session -maxdepth 2 -name "T1w*.nii*")
    sessions+=$( echo "$session " )
    T1ws+=$(echo "$T1w ")
 done < $txt
else
for T1w in `find $dirData -maxdepth 3 -name "T1w*.nii*"`
 do
    T1ws+=($T1w)
    IFS="/" read -r -a splitInitS <<< "$T1w"
    toStart2=${#splitInitS[@]}
    toLookFor=$((${toStart2}-2))
    session=$(echo $T1w | cut -d'/' -f1-${toLookFor})
    sessions+=$( echo "$session " )
 done
fi
counter=0
IFS=" " read -r -a listSessions <<< "$sessions"
IFS=" " read -r -a listT1ws <<< "$T1ws"
# lines 36 to end prepare and launch the pipeline
for session in "${listSessions[@]}"
 do
      ccase_small=$(echo "$ccase" | tr '[:upper:]' '[:lower:]')
      nipype_folderI="${DWIname}_pipe_${ccase_small}"
      nipype_folderF="${DWIname}_pipe_${ccase_small}_smoothing_${smooth}"
      outDir="$session/${nipype_folderF}"
      probMask="$session/${nipype_folderI}/DATASINK/DWIspace/WM_parcellation/bundle_segmentations.nii.gz"
      binMask="$session/${nipype_folderI}/DATASINK/DWIspace/WM_parcellation/bundle_segmentations_binary.nii.gz"
      mask="$session/${nipype_folderI}/DATASINK/DWIspace/masks/${dtiMaskName}"
      if [ "$ccase" == "LPCA_Gibbs_Eddy" ] || [ "$ccase" == "Eddy" ] || [ "$ccase" == "Gibbs_Eddy" ] || [ "$ccase" == "LPCA_Eddy" ] || [ "$ccase" == "LPCA_Gibbs_Eddy_noTopup" ] || [ "$ccase" == "Eddy_noTopup" ] || [ "$ccase" == "Gibbs_Eddy_noTopup" ] || [ "$ccase" == "LPCA_Eddy_noTopup" ] || [ "$ccase" == "full_length" ]; then
      bval="$session/${nipype_folderI}/DATASINK/DWIspace/dwi_proc/DWI_corrected.bval"
      bvec="$session/${nipype_folderI}/DATASINK/DWIspace/dwi_proc/DWI_corrected.bvec" 
     else
          bval=$(find $session -maxdepth 2 -name "*${DWIname}.bval")
          bvec=$(find $session -maxdepth 2 -name "*${DWIname}.bvec")
     fi
      DWIfiles=$session/${nipype_folderI}/DATASINK/DWIspace/dwi_proc
      if [ "$ccase" == "LPCA" ]; then
         DWI=$(find $DWIfiles -maxdepth 1 -name "DWI_denoised.nii*")
      elif [ "$ccase" == "LPCA_Gibbs" ] || [ "$ccase" == "Gibbs" ]; then
         DWI=$(find $DWIfiles -maxdepth 1 -name "DWI_*gibbs.nii*")
      else
         DWI=$(find $DWIfiles -maxdepth 1 -name "DWI_*topup_correct.nii*")
         #DWI=$(find $DWIfiles -maxdepth 1 -name "DWI_*eddy.nii*")
      fi
      if [ -f "$DWI" ] && [ -f "$bval" ] && [ -f "$bvec" ] && [ -f "$mask" ] && [ -f "$probMask" ] && [ -f "$binMask" ]; then
      if [ "$smooth" == "0.00" ]; then
          python ${scripts}/dwi_pipe_nonsmooth_tract_metrics.py -o $outDir -c $DWIname -d $DWI -b $bval -r $bvec -e $mask -k $smooth -p $probMask -m $binMask
       else
          python ${scripts}/dwi_pipe_smooth_tract_metrics.py -o $outDir -c $DWIname -d $DWI -b $bval -r $bvec -e $mask -k $smooth -p $probMask -m $binMask
       fi
      #rm -rf $dirOut/diffusion_dt_manualMask
       fi
 #   echo "----------------------------------------------"
#   fi
    count=$(($count+1))
 done

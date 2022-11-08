#! /bin/bash
# lines 3 to 16 define the initial values of the variables
scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
dirData=$1
IFS="/" read -r -a splitInit <<< "$dirData"
toStart=${#splitInit[@]}
DWIname=$2
anatBrainMaskName=$3
ccase=$4
dirAcqp=$5
DWIrevName=$6
dwiMaskName=$7
txt=$8
t1="T1w"
count=1
sessions=()
# lines 18 to 38 create the list of sessions/subjsects that are about to be processed by the pipeline
if [ $txt != "none" ]; then
  while IFS='' read -r line || [[ -n "$line" ]]; do
    session=$line
    T1w=$(find $session -maxdepth 2 -name "*${t1}.nii*")
    sessions+=$( echo "$session " )
    T1ws+=$(echo "$T1w ")
 done < $txt
else
for T1w in `find $dirData -maxdepth 3 -name "*${t1}.nii*"`
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
      cd $scripts
      cd ../anat_pipe/Atlases
      templatesDir=$(pwd)
      echo "Session: $session"
      T1=$session/anat_pipe_using_${t1}/DATASINK/N4_biasCorrected/T1_N4.nii.gz
      anatMask=$session/anat_pipe_using_${t1}/DATASINK/T1space/masks_atlases/${anatBrainMaskName}
      template=$templatesDir/CamCAN/CamCAN652_T1w_syn6.nii.gz
      deform=$session/anat_pipe_using_${t1}/DATASINK/CamCANspace/antsSyn_Composite.h5
      DWI=$(find $session -maxdepth 2 -name "*${DWIname}.nii*")
      bzero=$(find $session -maxdepth 2 -name "*${DWIrevName}.nii*")
      bval=$(find $session -maxdepth 2 -name "*${DWIname}.bval")
      bvec=$(find $session -maxdepth 2 -name "*${DWIname}.bvec")
      if [[ "$dwiMaskName" != "none" ]]; then
            dwi_mask=${dwiMaskName}
      fi
      if [ -f "$DWI" ] && [ -f "$bval" ] && [ -f "$bvec" ] && [ -f "$anatMask" ] && [ -f "$T1" ] && [ -f "$template" ] && [ -f "$deform" ]; then
       numVols=$(fslval $DWI dim4)
       acqp=$(ls ${dirAcqp})
       echo "$session+$numVols+$acqp" >> $session/dwi/info_on_${DWIname}_sessions.csv
       cd $session
      dirSlurms=$(pwd)
      if [ -f "$bzero" ]; then
         bzeroF=$bzero
      else
         bzeroF=None
      fi
      if [ -f "${dwi_mask}" ]; then
         dm=${dwi_mask}
      else
         dm=None
      fi
      if [ "$ccase" == "LPCA_Gibbs_Eddy" ]; then
              python ${scripts}/dwi_pipe_lpca_gibbs_eddy.py -o $session -c $DWIname -d $DWI -b $bval -r $bvec -e $dm -p $bzeroF -s $T1 -m $anatMask -a $acqp
      elif [ "$ccase" == "LPCA_Gibbs" ]; then
              python ${scripts}/dwi_pipe_lpca_gibbs.py -o $session -c $DWIname -d $DWI -b $bval -r $bvec -e $dm -p $bzeroF -s $T1 -m $anatMask -a $acqp
      elif [ "$ccase" == "LPCA" ]; then
              python ${scripts}/dwi_pipe_lpca.py -o $session -c $DWIname -d $DWI -b $bval -r $bvec -e $dm -p $bzeroF -s $T1 -m $anatMask -a $acqp
      elif [ "$ccase" == "LPCA_Eddy" ]; then
              python ${scripts}/dwi_pipe_lpca_eddy.py -o $session -c $DWIname -d $DWI -b $bval -r $bvec -e $dm -p $bzeroF -s $T1 -m $anatMask -a $acqp
      elif [ "$ccase" == "Gibbs_Eddy" ]; then
              python ${scripts}/dwi_pipe_gibbs_eddy.py -o $session -c $DWIname -d $DWI -b $bval -r $bvec -e $dm -p $bzeroF -s $T1 -m $anatMask -a $acqp
      elif [ "$ccase" == "Gibbs" ]; then
              python ${scripts}/dwi_pipe_gibbs.py -o $session -c $DWIname -d $DWI -b $bval -r $bvec -e $dm -p $bzeroF -s $T1 -m $anatMask -a $acqp
      elif [ "$ccase" == "Eddy" ]; then
              python ${scripts}/dwi_pipe_eddy.py -o $session -c $DWIname -d $DWI -b $bval -r $bvec -e $dm -p $bzeroF -s $T1 -m $anatMask -a $acqp
      elif [ "$ccase" == "full_length" ]; then
              python ${scripts}/dwi_pipe_full_length.py -o $session -c $DWIname -d $DWI -b $bval -r $bvec -e $dm -p $bzeroF -s $T1 -m $anatMask -t $template -n $deform -a $acqp
      fi
      fi
    count=$(($count+1))
 done

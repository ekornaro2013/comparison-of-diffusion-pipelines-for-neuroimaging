FROM nipy/mindboggle
COPY ./anat_pipe /anat_pipe
COPY ./dwi_pipe /dwi_pipe
RUN pip install scipy && pip install matplotlib && pip install nibabel && pip install nipype
CMD echo "This is a framework to process structural MRI and DTI or DKI data, designed by Evgenios N. Kornaropoulos" 

import os
import glob
import shutil
import nipype.interfaces.fsl as fsl

# folder structures are top_folder/group_folder/subject_folder/subject_nifti_files
def skull_strip_folder(input_folder, output_folder):
    for group_folder in glob.glob(os.path.join(input_folder, '*')):
        output_group_folder = os.path.join(output_folder, os.path.basename(group_folder))
        os.makedirs(output_group_folder)
        #print(output_group_folder)
        for subject_folder in glob.glob(os.path.join(group_folder, '*')):
            output_subject_folder = os.path.join(output_group_folder, os.path.basename(subject_folder))
            os.makedirs(output_subject_folder)
            print(output_subject_folder)
            for subject in glob.glob(os.path.join(subject_folder, '*')):
                for modality in ['t1', 't2', 'flair']: # skull strip these files
                    if modality in os.path.basename(subject):
                        output_subject = os.path.join(output_subject_folder, modality + '.nii.gz')
                        print(output_subject)
                        skull_strip(subject, output_subject)
                for modality in ['tu', 'truth']: # don't skull strip truth
                    if modality in os.path.basename(subject):
                        output_subject = os.path.join(output_subject_folder, modality + '.nii.gz')
                        shutil.copyfile(subject, output_subject)



def skull_strip(input_file, output_file, frac = 0.4):
    print(input_file)
    btr = fsl.BET()
    btr.inputs.in_file = input_file
    btr.inputs.frac = frac
    btr.inputs.out_file = output_file
    ressult = btr.run()

if __name__ == "__main__":
    print("skull stripping: ")
    #skull_strip('/media/mingrui/960EVO/datasets/tiantan/skullstrip_test/t2.nii.gz',
    #            '/media/mingrui/960EVO/datasets/tiantan/skullstrip_test/skull_stripped35.nii.gz',
    #            0.4)

    #skull_strip_folder('/media/mingrui/960EVO/datasets/tiantan/2017-11/tiantan_preprocessed', '/media/mingrui/960EVO/datasets/tiantan/2017-11/tiantan_skull_strip')
    #skull_strip_folder('/media/mingrui/960EVO/datasets/tiantan/2017-12/tiantan_preprocessed', '/media/mingrui/960EVO/datasets/tiantan/2017-12/tiantan_skull_strip')

    #skull_strip_folder('/media/mingrui/960EVO/datasets/tiantan/2017-12-TCGA-tiantan-IDH/tcga_t2_preprocessed', '/media/mingrui/960EVO/datasets/tiantan/2017-12-TCGA-tiantan-IDH/tcga_t2_skullstrip')


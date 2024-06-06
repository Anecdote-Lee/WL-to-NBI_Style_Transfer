import shutil
import os


def name_change(path):
    folder_list = ['Colorjitter', 'Flip', 'Rotation']
    for img in os.listdir(path+"/Colorjitter"):
        shutil.move(path + "/Colorjitter/" + img, path +"/cj_" + img)
    for img in os.listdir(path+"/Flip"):
        shutil.move(path + "/Flip/" + img, path +"/fl_" + img)        
    for img in os.listdir(path+"/Rotation"):
        shutil.move(path + "/Rotation/" + img, path +"/rt_" + img)
def main():

    path = "/home/jslee/uvcgan2/data/aug_crop_colonoscopic_resized_lanczos/train"
    name_change(path+"/nbi")
    name_change(path+"/wl")




if __name__ == '__main__':
    main()

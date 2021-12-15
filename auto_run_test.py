import csv
import os
import  glob

def do_prediction():
    folders = glob.glob('data/test/*')
    for folder in folders:
        dst = "inference/" + folder.split("/")[-1]

        # Check whether the specified path exists or not
        isExist = os.path.exists(dst)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(dst)

        os.system("python detect.py --source {} --output {} --weights trained_model/weights_car5/best.pt --conf 0.5 --device cpu".format(folder, dst))


def generate_result():
    # generate submission
    files = glob.glob('inference/*/*.jpg')
    sub_filename = "submission.csv"
    data = []
    one = 0
    zero = 0
    two = 0
    for file in files:
        name = file.replace("inference/", "")
        name = name.replace("_image.jpg", "")

        txt_path = file.replace(".jpg", ".txt")
        obj = []

        if os.path.exists(txt_path):
            content = open(txt_path)
            for line in content:
                obj.append(int(line.split(" ")[0]))
            print(obj)
            if 1 in obj:
                data.append([name, 1])
                one += 1
            elif 2 in obj:
                data.append([name, 2])
                two += 1
            else:
                data.append([name, 2])
                zero += 1
        else:
            data.append([name, 2])
            zero += 1
    print(zero, one, two)
    fields = ["guid/image", "label"]
    with open(sub_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(data)

    # print(file)
        # print(name)



if __name__ == '__main__':

    # prediction
    do_prediction()
    generate_result()



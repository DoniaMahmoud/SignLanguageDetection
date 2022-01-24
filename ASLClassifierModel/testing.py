import os
import requests

successTrials=0
numberoftrials=0

def checkdir(d):
    global numberoftrials
    global successTrials
    for file in os.listdir(d):
        url = 'https://api.talkingsigns.cf/uploader'
        filepath =os.path.join(d, file)
        classA = d.split("-")[0]
        classA = classA.split("\\")[1]
        files = {'file': open(filepath, 'rb').read()}
        response = requests.post(url, files=files)
        try:
            data = response.text
            numberoftrials = 1 + numberoftrials
            data= data.upper()
            classA=classA.upper()
            if(data.rstrip() == classA.rstrip()):
                print("Actual Class:", classA, "- Predicted Class:", data, "- PASS!")
                print("-----------------------------------------------")
                successTrials= successTrials + 1
            else:
                print("Actual Class:", classA, "- Predicted Class:", data, "- FAIL!")
                print("-----------------------------------------------")
        except:
            print("api testing failed")

def listdirs(rootdir):
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            checkdir(d)

rootdir = 'Classes' 
listdirs(rootdir)
print("Test Accuracy=",(successTrials/numberoftrials)*100, "%")

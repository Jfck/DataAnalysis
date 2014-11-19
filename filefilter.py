import os
class FileFilt:
    
    def __init__(self,extlist):
        self.fileList = []
        self.extList = extlist
        pass
    def FindFile(self,dirr,filtrate = 1):
        if os.path.exists(dirr) is True:
            for s in os.listdir(dirr):
                newDir = os.path.join(dirr,s)
                if os.path.isfile(newDir):
                    if filtrate:
                        if newDir and (os.path.splitext(newDir)[1] in self.extList):
                            self.fileList.append(newDir)
                    else:
                        self.fileList.append(newDir)

    def SetExt(self,extlist):
        self.fileList = []
        self.extList = extlist
        
 
if __name__ == "__main__":
    b = FileFilt([".jpg",".jpeg",".bmp",".png"])
    b.FindFile(dirr = "C:\Users\LJ\Desktop\803release\landsat_08m_10_Y5_v4_data")
    print(b.counter)
    for k in b.fileList:
        print k
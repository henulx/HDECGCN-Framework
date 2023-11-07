from HDECGCNdemo import MDTDSGCN
from HDECGCNdemo import HRACPCNN
import numpy as np

Resultb = HRACPCNN.each_acc
Resulta = MDTDSGCN.stat_res * 100

Resultb = np.array(Resultb)
Resulta = np.array(Resulta)

ResultAll = np.row_stack((Resulta,Resultb))
IP_Samples = [46,1428,830,237,483,730,28,478,20,972,2455,593,205,1265,386,93]
IP_Samples = np.array(IP_Samples)
IP_SumNum = 10249

# KSC_Samples = [761,243,256,252,161,229,105,431,520,404,419,503,927]
# KSC_Samples = np.array(KSC_Samples)
# KSC_SumNum = 5211

# UP_Samples = [6631,18649,2099,3064,1345,5029,1330,3682,947]
# UP_Samples = np.array(UP_Samples)
# UP_SumNum = 42776

def get_max_value(martix):
  '''
  Obtain the maximum value for each column in the matrix
  '''
  res_list=[]
  for j in range(len(martix[0])):
    one_list=[]
    for i in range(len(martix)):
      one_list.append(float(martix[i][j]))
    res_list.append(max(one_list))
  return res_list

TotalTime = HRACPCNN.train_time + MDTDSGCN.time

if __name__ == '__main__':
  CategoryAccuracy = get_max_value(ResultAll)
  AverageAccuracy = sum(CategoryAccuracy)/len(CategoryAccuracy)/100
  CategoryAccuracy = np.array(CategoryAccuracy)
  OverallAccuracy = np.dot(IP_Samples,CategoryAccuracy/100)/IP_SumNum
  #OverallAccuracy = np.dot(KSC_Samples, CategoryAccuracy / 100) / KSC_SumNum
  #OverallAccuracy = np.dot(UP_Samples, CategoryAccuracy/100) / UP_SumNum
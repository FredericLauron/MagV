import numpy as np
from operator import add
import json

with open("/home/ids/flauron-23/MagV/data/magv_06_unstructured_VariSampling_2/results/data_magv_06_unstructured_VariSampling_2_0.json", "r") as f:
    data_0 = json.load(f)

with open("/home/ids/flauron-23/MagV/data/magv_06_unstructured_VariSampling_2/results/data_magv_06_unstructured_VariSampling_2_5.json", "r") as f:
    data_5 = json.load(f)

with open("/home/ids/flauron-23/MagV/data/magv_06_unstructured_VariSampling_2/results/data_magv_06_unstructured_VariSampling_2_10.json", "r") as f:
    data_10 = json.load(f)







# #print(list( map(add, bpp["ours"],bpp["cheng2020"])))

# #print(list(set(bpp["ours"]) - set(bpp["cheng2020"])))
# # a=map(lambda x: x**2,list(set(bpp["ours"]) - set(bpp["cheng2020"])))
# # print(list(a))
# bpp_diff = (np.array(bpp["ours"],dtype=np.float64)-np.array(bpp["cheng2020"],dtype=np.float64))**2
# print(bpp_diff)
# psnr_diff = (np.array(psnr["ours"],dtype=np.float64)-np.array(psnr["cheng2020"],dtype=np.float64))**2
# print(psnr_diff)
# # mask = np.zeros_like(m, dtype = float)
# #np.isfinite(m, out = mask, where = m > 0.001)
# n = np.where(bpp_diff>0.001, 1.0, 0.0)
# print(n)
# m = np.where(psnr_diff>0.001, 1.0, 0.0)
# print(m)
# f=np.bitwise_or(n.astype(bool), m.astype(bool)).astype(float)
# print(f)
# # probs += n*0.1
# # probs = probs / probs.sum()


# print("probs",probs)
# # probs += f*0.1*bpp_diff*psnr_diff
# probs += f * 0.1 * (0.5 * bpp_diff + 0.5 * psnr_diff)
# print("probs update",probs)
# print("prob sum before nomalization",probs.sum())
# probs = probs / probs.sum()
# print("probs after normalization",probs)
# print("probs sum after normalization",probs.sum())

# sample = np.random.choice(arr, p=probs)

# # Todo sample from distance
# # probs += f*0.1*bpp_diff
# # probs += f*0.1*bpp_diff*psnr_diff

def adjust_sampling_distribution(bpp,psnr,probs):
    
    #Compute the squared diff of bpp and psnr
    bpp_diff = np.abs((np.array(bpp["ours"],dtype=np.float64)-np.array(bpp["cheng2020"],dtype=np.float64)))
    psnr_diff = np.abs((np.array(psnr["ours"],dtype=np.float64)-np.array(psnr["cheng2020"],dtype=np.float64)))
    print(psnr_diff)
    #Identify the indices where the diff is greater than a threshold
    #Currently set to 0.1, can be adjusted
    n = np.where(bpp_diff>0.1, 1.0, 0.0)
    print("bpp:",n)
    m = np.where(psnr_diff>0.9, 1.0, 0.0)
    print("psnr:",m)

    #Bitwise OR between the two masks    
    f=np.bitwise_or(n.astype(bool), m.astype(bool)).astype(float)
    print("f:",f)
    # Update the probs
    #probs += f*0.1*bpp_diff*psnr_diff
    #probs += f * 0.1 * (0.5 * bpp_diff + 0.5 * psnr_diff)
    #probs+=f*0.5*(bpp_diff+10*psnr_diff)
    probs+=f*0.2
    
    # Clipping for security
    probs = np.clip(probs, 1e-6, 1.0)

    # Normalize the probs
    probs = probs / probs.sum()

    print("probs after normalization",probs)
    print("probs sum after normalization",probs.sum())

    return probs

probs = np.ones(len(data_0["bpp"]["ours"])) / len(data_0["bpp"]["ours"]) 
print(probs)

for i in range(5):
    print(np.random.choice(np.arange(6), p=probs))

bpp ={"ours":data_0["bpp"]["ours"],"cheng2020":data_0["bpp"]["cheng2020"]}
psnr ={"ours":data_0["psnr"]["ours"],"cheng2020":data_0["psnr"]["cheng2020"]}

probs=adjust_sampling_distribution(bpp,psnr,probs)

for i in range(5):
    print(np.random.choice(np.arange(6), p=probs))

bpp ={"ours":data_5["bpp"]["ours"],"cheng2020":data_5["bpp"]["cheng2020"]}
psnr ={"ours":data_5["psnr"]["ours"],"cheng2020":data_5["psnr"]["cheng2020"]}

probs=adjust_sampling_distribution(bpp,psnr,probs)

for i in range(5):
    print(np.random.choice(np.arange(6), p=probs))

bpp ={"ours":data_10["bpp"]["ours"],"cheng2020":data_10["bpp"]["cheng2020"]}
psnr ={"ours":data_10["psnr"]["ours"],"cheng2020":data_10["psnr"]["cheng2020"]}

probs=adjust_sampling_distribution(bpp,psnr,probs)

for i in range(5):
    print(np.random.choice(np.arange(6), p=probs))


print(data_0["psnr"]["ours"])
print(data_0["psnr"]["cheng2020"])
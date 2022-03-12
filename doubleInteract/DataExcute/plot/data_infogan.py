import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
step = 50

result = []
with open('F:/2021spring/1erciyuangan/gan文章/paperdata/info with p 15/infogan_every_d_loss.txt', 'r') as f:
    for line in f:
        result.append(line.strip('\n') )
result = [float(x) for x in result]
result2=[]
for i, _ in enumerate(result[::step]):
    sub_list = result[i*10:] if (i+1)*10 > len(result) else result[i*10:(i+1)*10]
    result2.append(sum(sub_list)/float(len(sub_list)))
print(len(result2))

result_g = []
with open('F:/2021spring/1erciyuangan/gan文章/paperdata/info with p 15/infogan_every_g_loss.txt', 'r') as f:
    for line in f:
        result_g.append(line.strip('\n') )
result_g = [float(x) for x in result_g]
result3=[]
for i, _ in enumerate(result_g[::step]):
    sub_list = result_g[i*10:] if (i+1)*10 > len(result_g) else result_g[i*10:(i+1)*10]
    result3.append(sum(sub_list)/float(len(sub_list)))
print(len(result3))

result_d1= []
with open('F:/2021spring/1erciyuangan/gan文章/paperdata/info without p/infogan_every_d_loss.txt', 'r') as f:
    for line in f:
        result_d1.append(line.strip('\n') )
result_d1 = [float(x) for x in result_d1]
result4=[]

for i, _ in enumerate(result_d1[::step]):
    sub_list = result_d1[i*10:] if (i+1)*10 > len(result_d1) else result_d1[i*10:(i+1)*10]
    result4.append(sum(sub_list)/float(len(sub_list)))
print(len(result4))

result_g1 = []
with open('F:/2021spring/1erciyuangan/gan文章/paperdata/info without p/infogan_every_g_loss.txt', 'r') as f:
    for line in f:
        result_g1.append(line.strip('\n') )
result_g1 = [float(x) for x in result_g1]
result5=[]

for i, _ in enumerate(result_g1[::step]):
    sub_list = result_g1[i*10:] if (i+1)*10 > len(result_g1) else result_g1[i*10:(i+1)*10]
    result5.append(sum(sub_list)/float(len(sub_list)))
print(len(result5))

y1 = []
for i in range(len(result2)):
	    y1 += [i + 1]

plt.plot(y1, result2, '-r', ms=15, label='infogan with d')
plt.plot(y1, result3, '-g', ms=15, label='infogan with g')
plt.plot(y1, result4, '-b', ms=15, label='infogan without d')
plt.plot(y1, result5, '-k', ms=15, label='infogan without g')


plt.title(u"各种P下的结果对比")
plt.xlabel(u"Epoch")
plt.ylabel(u"Num")
plt.legend(loc=0, ncol=1)
plt.savefig('infogan.jpg')
plt.show()
import os
lo=open('./losss','r')
a=[]
for line in lo.readlines():
  a.append(line)
b=[ i.split(' ') for i in a]
loss1=[]
loss2=[]
for _ in b:
  try:
    loss1.append(float(_[0][:8]))
    loss2.append(float(_[1][:8]))
  except:
    continue
print(sum(loss1)/len(loss1))

print(sum(loss2)/len(loss2))
lo.close()

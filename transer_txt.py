import glob
import numpy as np
import h5py

Vid2Url = eval(open('url_to_vidId.txt').read())

f1 = open('./inv4sr14/inv4_feat_train.txt', 'w')
f2 = open('./inv4sr14/inv4_feat_val.txt', 'w')
f3 = open('./inv4sr14/inv4_feat_test.txt', 'w')

with h5py.File('./features/MSVD_InceptionV4_sr14.hdf5','r') as ff:
	for vid, c in enumerate(sorted(ff.keys())):
		for frame, feat in enumerate(ff[c]):
			#print("vid%d_frame_%d transform.." % (vid+1, frame+1))
			#print(feat.shape)
			d=Vid2Url[c]
			#print(d[3:])
			num=int(d[3:])
			d1 = "%s_frame_%d," % (d, frame + 1)
			#print(d1)
			d2 = ','.join(map(str, feat))
			data = d1 + d2 + '\n'
			if vid < 1200:
				f1.write(data)
				print("%s train, num %d"% (d, vid))
			elif vid < 1300:
				f2.write(data)
				print("%s val, num %d" %(d, vid))
			else:
				f3.write(data)
				print("%s test, num %d" %(d, vid))
			#print("shape %d" %(feat.shape))
f1.close()
f2.close()
f3.close()

print("FINISH")

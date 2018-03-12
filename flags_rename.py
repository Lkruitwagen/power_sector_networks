import glob
import os
import csv
from fuzzywuzzy import fuzz
import shutil

def main():
	fnames = glob.glob(os.path.join('flags_round', 'png', '*'))
	print fnames
	countries = []
	reader = csv.DictReader(open('country_ISO_update.csv','rt'))#, encoding='utf-8'))
	for line in reader:
		countries.append(line)

	print countries
	print fnames


	out_dict = []
	for f in fnames:
		ln_dict = {}
		ln_dict['fname'] = f
		ln_dict['LEFT'] = f.split('/')[-1][0:-4].upper()
		all_fuzz = [(fuzz.ratio(ln_dict['LEFT'],cap['Caps']),cap['Caps'],cap['ISO']) for cap in countries]
		ln_dict['RIGHT'] = max(all_fuzz, key=lambda x: x[0])[1]
		ln_dict['fuzz'] = max(all_fuzz, key=lambda x: x[0])[0]
		ln_dict['ISO'] = max(all_fuzz, key=lambda x: x[0])[2]
		print ln_dict['fuzz'], ln_dict['LEFT'], ln_dict['RIGHT']

		if ln_dict['fuzz']>0.9:
			shutil.copyfile(ln_dict['fname'], os.path.join('flags_round', 'ISO', ln_dict['ISO']+'.png'))
		out_dict.append(ln_dict)



if __name__ == "__main__":
	main()


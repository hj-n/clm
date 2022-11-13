import argparse
import numpy as np
import helpers as hp
import metric_run as mer
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run ablation Study', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--testtype", "-t", type=str, default="shift", help="set the type of the test ('shift' or 'card'), default is 'shift'")
parser.add_argument("--ablation", "-a", type=str, default="all", help='''set the ablation type used for the test 
'all': *, *_range, *_shift, *_btw
'original': *
'shift': only run *_shift 
'range': only run *_range 
'btw': only run *_btw 
''')
parser.add_argument("--measure", "-m", type=str, default="all", help='''set the measure used for the test
'ch': 'Calinski-Harabasz'
'sil': 'Silhouette'
'xbdb': 'Xie-Beni' + 'Davies-Bouldin'
'dunn': 'Dunn'
'ii': 'I Index',
''')
parser.add_argument("--function", "-f", type=str, default="all", help='''set the functionalities to run
'all': run both 'test' and 'plot'
'test': perform test (shift or card, set by --testype parameter)
'plot': plot the results (shift or card, set by --testype parameter). 
''')

args = parser.parse_args()

testtype_arg = args.testtype
ablation_arg = args.ablation
function_arg = args.function
measure_arg = args.measure


def generate_input_arrs(testtype_arg):
	dims = None
	sizes = None
	if testtype_arg == "shift":
		dims = np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
		sizes = hp.random_array_int(1000, 5000, 50000)
	elif testtype_arg == "card":
		dims = hp.random_array_int(1000, 2, 100)
		sizes = np.array([5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])

	return dims, sizes

def get_testtype_arr(testtype_arg):
	return {
		"shift": ["shift"],
		"card": ["card"],
		"all": ["shift", "card"]
	}[testtype_arg]

def get_measure_arr(ablation_arg, measure_arg):

	if measure_arg == "xbdb":
		measure_arg = "xb"

	measure_arr = {
		"all": [measure_arg, f"{measure_arg}_range", f"{measure_arg}_shift", f"{measure_arg}_btw"],
		"original": [measure_arg],
		"shift": [f"{measure_arg}_shift"],
		"range": [f"{measure_arg}_range"],
		"btw": [f"{measure_arg}_btw"]
	}[ablation_arg]

	if measure_arg == "xb":
		measure_arr.insert(0, "db")
	if measure_arg == "sil":
		measure_arr.remove("sil_range")
		measure_arr.remove("sil_shift")
	
	return measure_arr
	
def get_function_arr(function_arg):
	return {
		"all": ["test", "plot"],
		"test": ["test"],
		"plot": ["plot"]
	}[function_arg]

def run_test(testype, measure, dims, sizes):
	if testype == "shift":
		for i, dim in tqdm(enumerate(dims)):
			print("..........running test for dim =", dim, f"({i + 1}/{len(dims)})")
			scores = mer.run(measure, dim, sizes)
			hp.check_and_make(f"./results_shift/scores/{measure}")
			hp.save_json(scores.tolist(), f"./results_shift/scores/{measure}/{dim}.json")
	elif testype == "card":
		for i, size in tqdm(enumerate(sizes)):
			print("..........running test for size =", size, f"({i + 1}/{len(sizes)})")
			scores = mer.run(measure, dims, size)
			hp.check_and_make(f"./results_card/scores/{measure}")
			hp.save_json(scores.tolist(), f"./results_card/scores/{measure}/{size}.json")

def run_plot(testtype, measure, dims, sizes):
	keys = sizes if testtype == "card" else dims
	scores = hp.pairwise_smape(f"./results_{testtype}/scores", measure, keys)

	hp.plot_heatmap(f"./results_{testtype}/plots", testtype, measure, scores, keys)



def run_all(testtype_arg, ablation_arg, measure_arg, function_arg):
	np.random.seed(0)

	testtype_arr = get_testtype_arr(testtype_arg)
	measure_arr = get_measure_arr(ablation_arg, measure_arg)
	function_arr = get_function_arr(function_arg)

	for testtype in testtype_arr:
		dims, sizes = generate_input_arrs(testtype)
		print(f'...running {testtype} test')
		for measure in measure_arr:
			print(f'......running {measure} measure')
			for function in function_arr:
				if function == "test":
					run_test(testtype, measure, dims, sizes)
					print(f'..........test finished')
				elif function == "plot":
					run_plot(testtype, measure, dims, sizes)
					print(f'..........plotting finished')
				

run_all(testtype_arg, ablation_arg, measure_arg, function_arg)
	


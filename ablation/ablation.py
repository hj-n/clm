import argparse
import numpy as np
import helpers as hp
import metric_run as mer

parser = argparse.ArgumentParser(description='Run ablation Study', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--testtype", "-t", type=str, default="shift", help="set the type of the test ('shift' or 'card')")
parser.add_argument("--measure", "-m", type=str, default="all", help='''set the measure used for the test 
'all': all measures (CH, CH_range, CH_shift, CH_btw) 
'ch': only run CH 
'shift': only run CH_shift 
'range': only run CH_range 
'btw': only run CH_btw 
''')
parser.add_argument("--function", "-f", type=str, default="all", help='''set the functionalities to run
'all': run both 'test' and 'plot'
'test': perform test (shift or card, set by --testype parameter)
'plot': plot the results (shift or card, set by --testype parameter). 
''')

args = parser.parse_args()

testtype_arg = args.testtype
measure_arg = args.measure
function_arg = args.function


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

def get_measure_arr(measure_arg):
	return {
		"all": ["CH", "CH_range", "CH_shift", "CH_btw"],
		"ch": ["CH"],
		"shift": ["CH_shift"],
		"range": ["CH_range"],
		"btw": ["CH_btw"]
	}[measure_arg]

def get_function_arr(function_arg):
	return {
		"all": ["test", "plot"],
		"test": ["test"],
		"plot": ["plot"]
	}[function_arg]

def run_test(testype, measure, dims, sizes):
	if testype == "shift":
		for dim in dims:
			print("..........running test for dim =", dim)
			scores = mer.run(measure, dim, sizes)
			hp.save_json(scores.tolist(), f"./results_shift/scores/{measure}_{dim}.json")
	elif testype == "card":
		for size in sizes:
			print("..........running test for size =", size)
			scores = mer.run(measure, size, dims)
			hp.save_json(scores.tolist(), f"./results_card/scores/{measure}_{size}.json")

def run_plot(testtype, measure, dims, sizes):
	keys = sizes if testtype == "card" else dims
	scores = hp.pairwise_smape(f"./results_{testtype}/scores", measure, keys)
	hp.plot_heatmap(f"./results_{testtype}/plots", testtype, measure, scores, keys)



def run_all(testtype_arg, measure_arg, function_arg):
	np.random.seed(0)

	testtype_arr = get_testtype_arr(testtype_arg)
	measure_arr = get_measure_arr(measure_arg)
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
				

run_all(testtype_arg, measure_arg, function_arg)
	


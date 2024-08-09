import csv

def read_shapes_from_csv(filename: str) -> list[tuple[int, int, int, int, int, str]]:
    shapes = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes.append((
                int(row['BATCH']),
                int(row['NH']),
                int(row['SEQ_Q']),
                int(row['SEQ_KV']),
                int(row['D_HEAD']),
                str(row['dtype'])
            ))
    return shapes

def write_results_to_csv(results : list[tuple] | list[list] | list[dict], output_filename: str):
    if len(results) == 0:
        print('No valid results')
        return
    
    fieldnames = [
        'index', 
        'BATCH', 
        'NH', 
        'SEQ_Q', 
        'SEQ_KV', 
        'D_HEAD', 
        'dtype', 
        'mean_microseconds', 
        'arithmetic_intensity', 
        'tflops'
    ]

    with open(output_filename, 'w', newline='') as f:
        if isinstance(results[0], list) or isinstance(results[0], tuple):
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        elif isinstance(results[0], dict):
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
        else:
            print('Invalid result format')
            return
        
        for result in results:
            writer.writerow(result)
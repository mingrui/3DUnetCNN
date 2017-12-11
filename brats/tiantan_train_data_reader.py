import csv

def read_in_gene_file(gene_file_path):
    with open(gene_file_path, 'rb') as f:
        reader = csv.reader(f)
        target = list(reader)
    return target

def prepare_for_target():
    one_hot = [[1,0],[0,1]]
    targets = {}
    #sum_of_target = 0
    trainings = read_in_gene_file('data/training_tiantan.csv')
    for i in range(1,len(trainings)):
        targets[trainings[i][1]] = one_hot[int(trainings[i][6])]
        #sum_of_target += int(trainings[i][6])
    valids = read_in_gene_file('data/valid_tiantan.csv')
    for i in range(1,len(valids)):
        targets[valids[i][1]] = one_hot[int(valids[i][2])]
        #sum_of_target += int(valids[i][2])
    total_num = len(targets.keys())
    #print("total:"+str(total_num)+" "+" sum:"+str(sum_of_target))
    return targets

if __name__ == "__main__":
    targets = prepare_for_target()
    print(targets)
    print(len(targets.keys()))
    print(targets.values())

exit_code = "exit"
file_merge_data = []

while True:
    print("Add file name to merge, or type exit")
    file_name = input()
    if file_name == "exit":
        break
    with open(file_name) as file:
        for line in file:
            file_merge_data.append(line.rstrip())
        file.close()

merged_file = open("merge.txt", 'w')
for item in file_merge_data:
    merged_file.write("%s\n" % item)
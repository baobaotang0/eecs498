# 编写一种方法，对字符串数组进行排序，将所有变位词组合在一起。
# 变位词是指字母相同，但排列不同的字符串。
#
# 示例:
# 输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
# 输出:
# [
#   ["ate","eat","tea"],
#   ["nat","tan"],
#   ["bat"]
# ]
#
# 说明：
# 所有输入均为小写字母。
# 不考虑答案输出的顺序。


if __name__ == '__main__':
    str_set_ans = []
    pbset = ["eatt", "tea", "tan", "ate", "nat", "bat"]
    for i in range(len(pbset)):
        word = pbset[i]
        local_dict = {}
        for alpha in word:
            if alpha not in local_dict.keys():
                local_dict[alpha] = 1
            else:
                local_dict[alpha] += 1
        str_set_ans.append(local_dict)
    print(str_set_ans)
    ans = []
    used = []
    for i in range(len(str_set_ans)):
        if i in used:
            continue
        ans.append([pbset[i]])
        for j in range(min(len(str_set_ans)-1, i+1), len(str_set_ans)):
            if j in used:
                continue
            if str_set_ans[i] == str_set_ans[j]:
                ans[-1].append(pbset[j])
                used.append(j)
    print(ans)







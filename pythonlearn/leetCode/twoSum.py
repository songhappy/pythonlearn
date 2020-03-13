class Solution(object):
    def twoSum1(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dict = {}
        for i in range(len(nums)):
            dict[nums[i]] = i
        for i in range(len(nums)):
            diff = target - nums[i]
            if (diff in dict.keys() and dict[diff] != i):  # make sure it's not itself
                return [i, dict[diff]]
        return []

    def twoSum(self, nums, target):
        dict = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in dict.keys():
                return [i, dict[diff]]
            dict[nums[i]] = i
        return []
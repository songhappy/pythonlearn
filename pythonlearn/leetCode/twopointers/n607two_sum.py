class TwoSum:
    """
    @param number: An integer
    @return: nothing
    """

    def __init__(self):
        self.counter = {}

    def add(self, number):
        self.counter[number] = self.counter.get(number, 0) + 1

    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """

    def find(self, value):
        for ele in self.counter:
            if ele == value - ele:
                return self.counter[ele] >= 2
            elif value - ele in self.counter:
                return True

        return False

#608 sorted, does not care about duplicates
class Solution1:
    """
    @param nums: an array of Integer
    @param target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """

    def twoSum(self, nums, target):
        # write your code here
        left, right = 0, len(nums) - 1
        while left < right:
            sumtwo = nums[left] + nums[right]
            if sumtwo == target:
                return [left + 1, right + 1]
            elif sumtwo > target:
                right -= 1
            else:
                left += 1
        return []

#587, only two pointers
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """

    def twoSum6(self, nums, target):
        pairs = []
        nums.sort()

        left, right = 0, len(nums) - 1
        while left < right:
            sumtwo = nums[left] + nums[right]
            if sumtwo == target:
                pairs.append((nums[left], nums[right]))
                left += 1
                right -= 1
            elif sumtwo > target:
                right -= 1
            else:
                left += 1
        return len(list(set(pairs)))  #remove duplicate

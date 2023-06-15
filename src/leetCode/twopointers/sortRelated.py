# all sorts
import heapq


class Solution1:
    def heapSort1(self, A):
        tmp = []
        for item in A:
            heapq.heappush(tmp, item)
        return [heapq.heappop(tmp) for i in range(len(tmp))]

    def heapSort2(self, A):
        heapq.heapify(A)
        return [heapq.heappop(A) for i in range(len(A))]

    def heapSort(self, A):
        def heapify(A, n, i):  # 就是在 <n 的范围内，在i这个节点上的数是最大的
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2
            if l < n and A[largest] < A[l]:
                largest = l
            if r < n and A[largest] < A[r]:
                largest = r
            if largest != i:
                A[i], A[largest] = A[largest], A[i]
                heapify(A, n, largest)

        n = len(A)
        for i in range(n // 2, -1, -1):  # build 一个max heap, 完事后0的位置上是最大值
            heapify(A, n, i)

        for i in range(n - 1, -1, -1):
            A[0], A[i], = A[i], A[0]  # 把最大的这个值swap 到i 这个位置。 heapify 的时候 只在下标<i的范围内swift down
            heapify(A, i, 0)  # 又把下标<i的范围中，最大值丢到0
        return A

    def insertSort(self, A):
        for i in range(1, len(
            A)):  # 外层loop 从1 开始，每一个拿出来跟它前面所有的值比较，如果 <= 它前面的值，需要往前挪，直到位置，前面的比key 小，后面的>=key
            key = A[i]
            j = i - 1
            for j in range(i - 1, -1, -1):  # 内层loop 从 i 向前看
                if key <= A[j]:  # >=key 的通通往后挪一位，j往前挪一位
                    A[j + 1] = A[j]
                    j -= 1
            A[j + 1] = key  # j+1就是key 的位置

    def bubbleSort(self, A):  # loop 从1 开始，每一个拿出来跟它前面的值比较，如果 <=, 就swap
        for i in range(1, len(A)):
            for j in range(i, 0, -1):
                if A[j] <= A[j - 1]:
                    A[j], A[j - 1] = A[j - 1], A[j]

    def sortIntegers(self, A):
        self.quickSort(A, 0, len(A) - 1)

    def quickSort(self, A, start, end):
        if start >= end:
            return

        i, j = start, end
        # key point 1: pivot is the value, not the index
        pivot = A[(i + j) // 2]
        # key point 2: every time you compare left & right, it should be
        # left <= right not left < right
        while i <= j:
            while i <= j and A[i] < pivot:  # 如果左边的都<pivot, 一直走，直到 => pivot
                i += 1
            while i <= j and A[j] > pivot:  # 如果右边的>pivot, 一直走，直到 <= pivot
                j -= 1

            if i <= j:  # 现在A[i] >= pivot, A[j]<= pivot, swap
                A[i], A[j] = A[j], A[i]
                i += 1
                j -= 1
        # 这一轮循环出来的时候，j+1的位置放了pivot值
        # partition 就是找到原来pivot 这个值的位置 j + 1, 使左边的值<它，右边的>它，等于它的去哪里了？左边先遇到的话就去了右边，右边先遇到就丢到了左边。
        self.quickSort(A, start, j)
        self.quickSort(A, i, end)

    def mergesort(self, seq):
        if len(seq) <= 1:
            return seq
        mid = len(seq) // 2  # 将列表分成更小的两个列表
        # 分别对左右两个列表进行处理，分别返回两个排序好的列表
        left = self.mergesort(seq[:mid])
        right = self.mergesort(seq[mid:])
        # 对排序好的两个列表合并，产生一个新的排序好的列表
        return self.merge(left, right)

    def merge(self, left, right):
        """合并两个已排序好的列表，产生一个新的已排序好的列表"""
        result = []  # 新的已排序好的列表
        i = 0  # 下标
        j = 0
        # 对两个列表中的元素 两两对比。
        # 将最小的元素，放到result中，并对当前列表下标加1
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result += left[i:]
        result += right[j:]
        return result


# merge sort00
class Solution5:
    def SortIntegeters(self, A):
        if not A:
            return A
        tmp = [0] * len(A)
        self.merge_sort(A, 0, len(A) - 1, tmp)

    def merge_sort(self, A, start, end, tmp):
        if start >= end:
            return
        self.merge_sort(A, start, (start + end) // 2, tmp)
        self.merge_sort(A, (start + end) // 2, end, tmp)
        self.merge(A, start, end, tmp)

    def merge(self, A, start, end, tmp):
        i, j = start, end
        mid = (start + end) // 2
        k = start
        while i <= mid and j <= end:
            if A[i] < A[j]:
                tmp[k] = A[i]
                i += 1
            else:
                tmp[k] = A[j]
                j += 1
            k += 1
        while i <= mid:
            tmp[k] = A[i]
            i += 1
            k += 1
        while j <= end:
            tmp[k] = A[j]
            j += 1
            k += 1

        for l in range(start, end + 1):
            A[l] = tmp[l]


if __name__ == '__main__':
    import random

    A = [random.randint(0, 100) for i in range(10)]
    print(A)
    sort = Solution1()
    # quick sort
    sort.sortIntegers(A)
    print(A)

    # bubble sort
    A = [random.randint(0, 100) for i in range(10)]
    print(A)
    sort.bubbleSort(A)
    print(A)

    # insert sort
    A = [random.randint(0, 100) for i in range(10)]
    print("****")
    print(A)
    sort.insertSort(A)
    print(A)

    # heap sort
    A = [random.randint(0, 100) for i in range(10)]
    print(A)
    sort.heapSort1(A)
    print(A)

    # heap sort
    A = [random.randint(0, 100) for i in range(10)]
    print("****")
    print(A)
    sort.heapSort2(A)
    print(A)
    A = [random.randint(0, 100) for i in range(10)]
    print("****")
    print(A)
    A = sort.heapSort(A)
    print(A)

/**
Author: David GC
Date: 30 November 2023
Problem : Leetcode 
 */

var happyNumber = function (n) {
  // create a set to keep the track if a cycle is detected
  let visitedNumber = new Set();
  function recurse(num) {
    //base condition to exit the recursion or the number is a happy number
    if (num == 1) {
      return true;
    }

    if (visitedNumber.has(num)) {
      return false;
    }

    visitedNumber.add(num);

    let squaredSum = 0;
    let arrayNum = [...num.toString()].map(Number);

    for (let i = 0; i < arrayNum.length; i++) {
      squaredSum += arrayNum[i] * arrayNum[i];
    }

    return recurse(squaredSum);
  }

  return recurse(n);
};

// console.log(happyNumber(2));

// Leetcode problem 112: Path sum

var pathSum = function (root, targetSum) {
  //dfs recursion
  function recurse(root, currSum) {
    //base case
    if (root === null) {
      return false;
    }

    currSum += root.val;

    if (!root.left && !root.right) {
      //if there is no left and right leaf node then we compare the current sum to the target sum
      return currSum === targetSum;
    }

    return recurse(root.left, currSum) || recurse(root.right, currSum);
  }

  return recurse(root, 0);
};

// Leetcode problem 67: Add binary

var addBinary = function (a, b) {
  //create a result var as a string
  let result = "";

  let i = a.length - 1;
  let j = b.length - 1;
  let carry = 0;

  //loop through the binary digits from right to left
  while (i >= 0 || j >= 0) {
    let sum = carry;
    //sum of two bits
    if (i >= 0) {
      //If there are more bits in a, add the current bit to the sum
      sum += a[i--] - "0";
    }

    if (j >= 0) {
      //If there are more bits in a, add the current bit to the sum
      sum += b[j--] - "0";
    }

    //Add the bit to the result
    result = (sum % 2) + result;

    //modify carry
    carry = parseInt(sum / 2);
  }

  //check for the final time if there is any carry
  if (carry > 0) {
    result = 1 + result;
  }

  return result;
};

addBinary("11", "1");

//problem 387: First unique character in string

var firstUniqChar = function (s) {
  //Initalize a map to keep count of characters
  const map = new Map();
  let count = 0;
  for (let i = 0; i < s.length; i++) {
    if (!map.has(s[i])) {
      map.set(s[i], count);
    } else {
      map.set(s[i], count + 1);
    }
  }

  //Find the index of the first character with count 0
  for (let i = 0; i < s.length; i++) {
    if (map.get(s[i] === 0)) {
      return i;
    }
  }

  //If no characters with count 0 is found, return -1
  return -1;
};

firstUniqChar("loveleetcode");

var findDuplicate = function (nums) {
  //detect if there's a cycle
  let tortoise = nums[0];
  let hare = nums[0];

  do {
    tortoise = nums[tortoise];
    hare = nums[nums[hare]];
  } while (tortoise != hare);

  //find the entrance to the cycle (duplicate number)
  tortoise = nums[0];

  while (tortoise != hare) {
    tortoise = nums[tortoise];
    hare = nums[hare];
  }

  return tortoise;
};

findDuplicate([1, 3, 4, 2, 2]);

// var productOfArrayExceptSelf = function (nums) {
//   let start = 1;
//   let res = [];

//   for (let i = 0; i < nums.length; i++) {
//     res.push(start);
//     start = start * nums[i];
//     // 1, 1, 2, 6
//   }

//   let start2 = 1;

//   for (let i = nums.length - 1; i >= 0; i--) {
//     res[i] = start2 * res[i];

//     start2 = start2 * nums[i];

//   }

//   return res;
// };

// productOfArrayExceptSelf([1, 2, 3, 4]);

function productExceptSelf(nums) {
  const n = nums.length;

  // Initialize two arrays to store products to the left and right of each element.
  const leftProducts = new Array(n).fill(1);
  const rightProducts = new Array(n).fill(1);

  // Calculate products to the left of each element.
  let leftProduct = 1;
  for (let i = 1; i < n; i++) {
    leftProduct *= nums[i - 1];
    leftProducts[i] = leftProduct;
  }

  // Calculate products to the right of each element.
  let rightProduct = 1;
  for (let i = n - 2; i >= 0; i--) {
    rightProduct *= nums[i + 1];
    rightProducts[i] = rightProduct;
  }

  // Multiply corresponding left and right products to get the final result.
  const result = [];
  for (let i = 0; i < n; i++) {
    result[i] = leftProducts[i] * rightProducts[i];
  }

  return result;
}

// Example usage:
const nums = [1, 2, 3, 4];
const result = productExceptSelf(nums);

var mergeIntervals = function (intervals) {
  let start = 0;
  let end = 1;

  //sort array to compare the previous and current interval value
  intervals = intervals.sort((a, b) => a[start] - b[start]);

  //assign the previous variable
  let previous = intervals[0];
  //result variable to store the output
  let result = [previous];

  for (let current of intervals) {
    if (previous[end] >= current[start]) {
      //get the max value from the end of prev and current
      previous[end] = Math.max(previous[end], current[end]);
    } else {
      result.push(current);

      //assign previous as current
      previous = current;
    }
  }

  return result;
};

mergeIntervals([
  [1, 3],
  [2, 6],
  [8, 10],
  [10, 12],
]);

var letterCombinations = function (digits) {
  const alpha = {
    2: "abc",
    3: "def",
    4: "ghi",
    5: "jkl",
    6: "mno",
    7: "pqrs",
    8: "tuv",
    9: "wxyz",
  };

  const result = [];

  //dfs recursive helper
  const dfs = (i, digits, slate) => {
    //base case
    if (i === digits.length) {
      result.push(slate.join(""));

      return;
    }

    //dfs recursive case
    let chars = alpha[digits[i]];

    for (let char of chars) {
      slate.push(char);
      dfs(i + 1, digits, slate);
      slate.pop();
    }
  };

  dfs(0, digits, []);

  return result;
};

letterCombinations("23");

var subarraySum = function (nums, k) {
  const map = new Map();
  let count = 0;
  let currentSum = 0;

  for (let num of nums) {
    currentSum += num;

    if (currentSum === k) {
      count++;
    }

    const complement = currentSum - k;
    if (map.has(complement)) {
      count += map.get(complement);
    }

    if (!map.has(currentSum)) {
      map.set(currentSum, 0);
    }

    map.set(currentSum, map.get(currentSum) + 1);
  }

  return count;
};

subarraySum([1, 2, 1, 2, 1], 3);

var maxProduct = function (nums) {
  let prevMax = nums[0];
  let prevMin = nums[0];
  let result = nums[0];

  for (let i = 1; i < nums.length; i++) {
    let currMax = Math.max(nums[i], nums[i] * prevMax, nums[i] * prevMin);
    let currMin = Math.min(nums[i], nums[i] * prevMin, nums[i] * prevMax);

    prevMax = currMax;
    prevMin = currMin;

    result = Math.max(result, prevMax);
  }

  return result;
};

maxProduct([2, 3, -2, 4]);

var permute = function (nums, arr = [], res = []) {
  //base case
  if (nums.length === 0) res.push([...arr]);

  for (let i = 0; i < nums.length; i++) {
    let rest = nums.filter((n, index) => index !== i);
    arr.push(nums[i]);
    permute(rest, arr, res);
    arr.pop();
  }

  return res;
};

permute([1, 2, 3]);

var mergeSortedArray = function (nums1, m, nums2, n) {
  let i = m - 1;
  let j = n - 1;
  let k = m + n - 1;

  while (i >= 0 && j >= 0) {
    if (nums1[i] > nums2[j]) {
      nums1[k] = nums1[i];
      i--;
    } else {
      nums1[k] = nums2[j];
      j--;
    }

    k--;
  }

  while (j >= 0) {
    nums1[k] = nums2[j];
    j--;
    k--;
  }

  return nums1;
};

mergeSortedArray([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3);

var combinationSum = function (candidates, target) {
  //assign a result variable to store the possible combinations
  let result = [];

  function dfs(index, currentVal, arr) {
    //base case
    if (currentVal < 0) return;
    if (currentVal === 0) {
      result.push([...arr]);
    }

    //iterate over arr and subtract target with the arr[i]
    for (let i = index; i < candidates.length; i++) {
      arr.push(candidates[i]);
      dfs(i, currentVal - candidates[i], arr);
      arr.pop();
    }
  }

  dfs(0, target, []);

  return result;
};

combinationSum([2, 3, 6, 7], 7);

var canJump = function (nums) {
  let target = nums.length - 1;

  for (let i = nums.length - 1; i >= 0; i--) {
    if (i + nums[i] >= target) {
      target = i;
    }
  }

  return target == 0;
};

canJump([3, 2, 1, 0, 4]);

var findKthLargest = function (nums, k) {
  nums.sort((a, b) => b - a);
  return nums[k];
};

findKthLargest([3, 2, 1, 5, 6, 4], 2);

var subsets = function (nums) {
  let result = [[]];

  var dfs = function (index, current) {
    for (let i = index; i < nums.length; i++) {
      current.push(nums[i]);
      result.push([...current]);

      dfs(i + 1, current);
      //backtrack
      current.pop();
    }
  };

  dfs(0, []);
  return result;
};

subsets([1, 2, 3]);

function TreeNode(val, left, right) {
  this.val = val === undefined ? 0 : val;
  this.left = left === undefined ? null : left;
  this.right = right === undefined ? null : right;
}

var inorderTraversal = function (root) {
  let arr = [];

  checkTree(root, arr);

  function checkTree(root, arr) {
    if (root == null) {
      return root;
    }

    checkTree(root.left, arr);
    arr.push(root.val);
    checkTree(root.right, arr);
  }
  return arr;
};

const isValidBST = function (root) {
  function recurse(root, min, max) {
    //base case
    if (root === null) return true;

    if (root.val >= max || root.val <= min) {
      return false;
    }

    return (
      recurse(root.left, min, root.val) && recurse(root.right, root.val, max)
    );
  }

  return recurse(root, -Infinity, Infinity);
};

const averageOfLevels = function (root) {
  let q = [root];
  let ans = [];

  while (q.length) {
    let qlen = q.length;
    let row = 0;

    for (let i = 0; i < qlen; i++) {
      let curr = q.shift();
      row += curr.val;

      if (curr.left) q.push(curr.left);
      if (curr.right) q.push(curr.right);
    }
    ans.push(row / qlen);
  }

  return ans;
};

const root = new TreeNode(3);
root.left = new TreeNode(9);
root.right = new TreeNode(20);
root.right.left = new TreeNode(15);
root.right.right = new TreeNode(7);

averageOfLevels(root);

var topKFrequent = function (nums, k) {
  let map = {};
  let bucket = [];
  let result = [];

  for (let i = 0; i < nums.length; i++) {
    if (!map[nums[i]]) {
      map[nums[i]] = 1;
    } else {
      map[nums[i]]++;
    }
  }

  for (let [num, freq] of Object.entries(map)) {
    if (!bucket[freq]) {
      bucket[freq] = new Set().add(num);
    } else {
      bucket[freq] = bucket[freq].add(num);
    }
  }

  for (let i = bucket.length; i >= 0; i--) {
    if (bucket[i]) result.push(...bucket[i]);
    if (result.length === k) break;
  }

  return result;
};

topKFrequent([1, 1, 1, 2, 2, 3], 2);

var sortColors = function (nums) {
  // let count0 = 0;
  // let count1 = 0;
  // let count2 = 0;
  // for (let i = 0; i < nums.length; i++) {
  //   if (nums[i] == 0) {
  //     count0++;
  //   } else if (nums[i] == 1) {
  //     count1++;
  //   } else {
  //     count2++;
  //   }
  // }

  // let i = 0;
  // while (count0 > 0) {
  //   nums[i] = 0;
  //   count0--;
  //   i++;
  // }

  // while (count1 > 0) {
  //   nums[i] = 1;
  //   count1--;
  //   i++;
  // }

  // while (count2 > 0) {
  //   nums[i] = 2;
  //   count2--;
  //   i++;
  // }

  // Another solution using the dutch national flag algorithm
  let low = 0,
    mid = 0;
  let high = nums.length - 1;

  for (let i = 0; i <= nums.length; i++) {
    if (nums[mid] === 0) {
      [nums[mid], nums[low]] = [nums[low], nums[mid]];
      mid++;
      low++;
    } else if (nums[mid] === 2) {
      [nums[mid], nums[high]] = [nums[high], nums[mid]];
      mid++;
      high--;
    } else {
      mid--;
    }
  }
  return nums;
};

sortColors([2, 0, 2, 1, 1, 0]);

var rotateImage = function (matrix) {
  //transpose the matrix
  for (let i = 0; i < matrix.length; i++) {
    for (let j = i; j < matrix.length; j++) {
      let temp = matrix[i][j];
      matrix[i][j] = matrix[j][i];
      matrix[j][i] = temp;
    }
  }

  //j loops till the length/2. if we go till length it will reverse back to original
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length / 2; j++) {
      let temp = matrix[i][j];
      //matrix.length - 1 - j to reverse the inwards matrix
      matrix[i][j] = matrix[i][matrix.length - 1 - j];
      matrix[i][matrix.length - 1 - j] = temp;
    }
  }

  return matrix;
};

rotateImage([
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
]);

var decodeString = function (s) {
  let charStack = [];
  let numStack = [];
  let currNum = 0;
  let currStr = "";

  for (let char of s) {
    if (!isNaN(char)) {
      //if char is 100 without currNum, each of the numbers would be push instead of 100
      currNum = currNum * 10 + parseInt(char);
    } else if (char == "[") {
      numStack.push(currNum);
      charStack.push(currStr);
      currNum = 0;
      currStr = "";
    } else if (char == "]") {
      let num = numStack.pop();
      let prevStr = charStack.pop();
      currStr = prevStr + currStr.repeat(num);
    } else {
      currStr += char;
    }
  }

  return currStr;
};

decodeString("100[c]2[b]");

var Trie = function () {
  this.root = {};
};

Trie.prototype.insert = function (word) {
  let node = this.root;

  for (let c of word) {
    if (node[c] == null) node[c] = {};
    node = node[c];
  }

  node.isWord = true;
};

Trie.prototype.traverse = function (word) {
  let node = this.root;

  for (let c of word) {
    node = node[c];
    if (node == null) return null;
  }

  return node;
};

Trie.prototype.search = function (word) {
  let node = this.traverse(word);

  return node !== null && node.isWord == true;
};

Trie.prototype.startsWith = function (prefix) {
  let node = this.traverse(prefix);
  return node !== null;
};

var obj = new Trie();
obj.insert("apple");
var word1 = obj.search("apple");
var word2 = obj.startsWith("apple");

var levelOrder = function (root) {
  if (root === null) return null;

  let res = [];
  let queue = [root];

  while (queue.length) {
    let levelArr = [];
    let levelSize = queue.length;

    while (levelSize) {
      let current = queue.shift();
      if (current.left) queue.push(current.left);
      if (current.right) queue.push(current.right);

      levelArr.push(current.val);
      levelSize--;
    }
    res.push(levelArr);
  }

  return res;
};

let node1 = new TreeNode(3);
node1.left = new TreeNode(9);
node1.right = new TreeNode(20);
node1.left.left = null;
node1.left.right = null;
node1.right.left = new TreeNode(15);
node1.right.right = new TreeNode(7);

var findAnagrams = function (s, p) {
  let map = {};

  for (let c of p) {
    map[c] = (map[c] || 0) + 1;
  }

  let rightPointer = 0,
    leftPointer = 0,
    count = 0;
  let res = [];

  while (rightPointer < s.length) {
    if (map[s[rightPointer]] > 0) count++;
    map[s[rightPointer]]--;
    rightPointer++;

    if (count === p.length) res.push(leftPointer);
    if (rightPointer - leftPointer === p.length) {
      if (map[s[leftPointer]] >= 0) count--;
      map[s[leftPointer]]++;
      leftPointer++;
    }
  }
  return res;
};

findAnagrams("cbaebabacd", "aabc");

var uniquePaths = function (m, n) {
  let dp = Array.from(Array(m), () => new Array(n));

  for (let i = 0; i < dp.length; i++) dp[i][0] = 1;
  for (let i = 0; i < dp[0].length; i++) dp[0][i] = 1;

  for (let i = 1; i < dp.length; i++) {
    for (let j = 1; j < dp[0].length; j++) {
      dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
    }
  }

  return dp[m - 1][n - 1];
};

uniquePaths(3, 7);

var rob = function (nums) {
  if (nums.length === 0) return 0;
  if (nums.length === 1) return nums[0];

  // Keep track of the max money we can make with x amount of houses available
  // dp[0] = max amount if we only have the first house to rob
  // dp[1] = max amount if we only have the first 2 houses to rob
  let dp = [nums[0], Math.max(nums[0], nums[1])];

  for (let i = 2; i < nums.length; i++) {
    // Compare current max with the previous max
    // Check if the money from the current house + max of 2 houses away is greater than the current max
    dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
  }
  return dp[nums.length - 1];
};

var coinChange = function (coins, amount) {
  let dp = new Array(amount + 1).fill(Infinity);

  //base case
  dp[0] = 0;

  for (let currAmount = 1; currAmount <= amount; currAmount++) {
    for (let coin of coins) {
      if (currAmount - coin >= 0) {
        dp[currAmount] = Math.min(dp[currAmount], 1 + dp[currAmount - coin]);
      }
    }
  }

  return dp[amount] > amount ? -1 : dp[amount];
};

coinChange([1, 2, 5], 11);

var courseSchedue = function (numCourses, prerequisites) {
  let adjList = {};
  let visited = new Set();

  for (let [a, b] of prerequisites) {
    if (!adjList[a]) {
      adjList[a] = [b];
    } else {
      adjList[a].push(b);
    }
  }

  function dfs(curr) {
    if (visited.has(curr)) return false;

    if (adjList[curr] == []) return true;

    visited.add(curr);

    if (adjList[curr]) {
      for (let neigh of adjList[curr]) {
        if (!dfs(neigh)) {
          return false;
        }
      }
    }

    visited.delete(curr);
    adjList[curr] = [];

    return true;
  }

  for (let key in adjList) {
    if (!dfs(key)) {
      return false;
    }
  }

  return true;
};

courseSchedue(2, [[1, 0]]);

var removeNthFromEnd = function (head, n) {
  let dummy = new ListNode(0);
  dummy.next = head;

  let left = dummy;
  let right = head;

  while (right && n > 0) {
    right = right.next;
    n -= 1;
  }

  while (right) {
    left = left.next;
    right = right.next;
  }

  left.next = left.next.next;
  return dummy.next;
};

const head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);
head.next.next.next = new ListNode(4);
head.next.next.next.next = new ListNode(5);

removeNthFromEnd(head, 4);

var addTwoNumbers = function (l1, l2) {
  let dummy = new ListNode(0);
  let head = dummy;

  let sum = 0;
  let carry = 0;

  while (l1 != null || l2 != null || sum != 0) {
    if (l1 != null) {
      sum += l1.val;
      l1 = l1.next;
    }

    if (l2 != null) {
      sum += l2.val;
      l2 = l2.next;
    }

    if (sum >= 10) {
      carry = 1;
      sum = sum - 10;
    }
    head.next = new ListNode(sum);
    head = head.next;
    sum = carry;
    carry = 0;
  }

  return dummy.next;
};

const l1 = new ListNode(2);
l1.next = new ListNode(4);
l1.next.next = new ListNode(3);

const l2 = new ListNode(5);
l2.next = new ListNode(6);
l2.next.next = new ListNode(4);

addTwoNumbers(l1, l2);

function ListNode(val, next) {
  this.val = val === undefined ? 0 : val;
  this.next = next === undefined ? null : next;
}

var mergeTwoLists = function (list1, list2) {
  let List = new ListNode(0);
  let head = List;

  while (list1 && list2) {
    if (list1.val >= list2.val) {
      List.next = list2;
      list2 = list2.next;
    } else {
      List.next = list1;
      list1 = list1.next;
    }

    List = List.next;
  }

  if (list1 != null) {
    List.next = list1;
  } else {
    List.next = list2;
  }

  return head.next;
};

const list1 = new ListNode(1);
list1.next = new ListNode(2);
list1.next.next = new ListNode(4);

const list2 = new ListNode(1);
list2.next = new ListNode(3);
list2.next.next = new ListNode(4);

mergeTwoLists(list1, list2);

var swapPairs = function (l1) {
  let dummy = new ListNode(0);
  dummy.next = l1;
  let prev = dummy;

  while (l1 && l1.next) {
    let pointer1 = l1;
    let pointer2 = l1.next;

    prev.next = pointer2;
    pointer1.next = pointer2.next;
    pointer2.next = pointer1;

    prev = pointer1;
    l1 = l1.next;
  }

  return dummy.next;
};

const swapPairsl1 = new ListNode(1);
swapPairsl1.next = new ListNode(2);
swapPairsl1.next.next = new ListNode(3);
swapPairsl1.next.next.next = new ListNode(4);

swapPairs(swapPairsl1);

var buildTree = function (preorder, inorder) {
  function recurse(pStart, pEnd, inStart, inEnd) {
    // Base case
    if (pStart > pEnd || inStart > inEnd) return null;

    let rootVal = preorder[pStart];
    let inIndex = inorder.indexOf(rootVal);
    let nLeft = inIndex - inStart;

    let root = new TreeNode(rootVal);

    root.left = recurse(pStart + 1, pStart + nLeft, inStart, inIndex - 1);
    root.right = recurse(pStart + 1 + nLeft, pEnd, inIndex + 1, inEnd);

    return root;
  }

  return recurse(0, preorder.length - 1, 0, inorder.length - 1);
};

// buildTree([3, 9, 20, 15, 7], [9, 3, 5, 20, 7]);

function sortList(head) {
  if (head === null || head.next === null) return head;

  // Step 1: Split the list into two halves
  const mid = getMid(head);
  const left = sortList(head);
  const right = sortList(mid);

  // Step 2: Merge the sorted lists
  return merge(left, right);
}

function getMid(head) {
  let prev = null;
  let slow = head;
  let fast = head;

  while (fast !== null && fast.next !== null) {
    prev = slow;
    slow = slow.next;
    fast = fast.next.next;
  }

  if (prev !== null) prev.next = null;

  return slow;
}

function merge(l1, l2) {
  const dummy = new ListNode();
  let current = dummy;

  while (l1 !== null && l2 !== null) {
    if (l1.val < l2.val) {
      current.next = l1;
      l1 = l1.next;
    } else {
      current.next = l2;
      l2 = l2.next;
    }
    current = current.next;
  }

  if (l1 !== null) current.next = l1;
  else current.next = l2;

  return dummy.next;
}

var LRUCache = function (capacity) {
  this.cache = new Map();
  this.capacity = capacity;
};

LRUCache.prototype.put = function (key, value) {
  if (this.cache.has(key)) {
    this.cache.delete(key);
  }

  this.cache.set(key, value);
  if (this.cache.size > this.capacity) {
    this.cache.delete(this.cache.keys().next().value);
  }
};

LRUCache.prototype.get = function (key) {
  if (!this.cache.has(key)) {
    return -1;
  }

  const v = this.cache.get(key);
  this.cache.delete(key);
  this.cache.set(key, v);

  return this.cache.get(key);
};

const cache = new LRUCache(2);
cache.put(1, 1);
cache.put(2, 2);
cache.get(1);
cache.get(2);
cache.put(4, 4);

var majorityElement = function (nums) {
  // let map = new Map();
  // let maxKey;
  // let maxValue = -Infinity;

  // for (let i = 0; i < nums.length; i++) {
  //   if (!map.has(nums[i])) {
  //     map.set(nums[i], 1);
  //   } else {
  //     map.set(nums[i], map.get(nums[i]) + 1);
  //   }
  // }

  // for (let [key, value] of map.entries()) {
  //   if (value > maxValue) {
  //     maxValue = value;
  //     maxKey = key;
  //   }
  // }

  // return maxKey;

  //boyer-moore majority vote algorithm

  let count = 1;
  let maxElement = nums[0];

  for (let i = 1; i < nums.length; i++) {
    if (count === 0) {
      maxElement = nums[i];
      count = 1;
    } else if (nums[i] == maxElement) {
      count++;
    } else {
      count--;
    }
  }

  return maxElement;
};

majorityElement([6, 5, 5]);

var palindromeLinkedList = function (head) {
  let fast = head;
  let slow = head;

  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  fast = head;
  slow = reverse(slow);

  while (slow) {
    if (fast.val !== slow.val) {
      return false;
    }

    slow = slow.next;
    fast = fast.next;
  }

  return true;

  function reverse(root) {
    let prev = null;

    while (root) {
      let ref = root.next;
      root.next = prev;
      prev = root;
      root = ref;
    }

    return prev;
  }
};

const list = new ListNode(1);
list.next = new ListNode(2);
list.next.next = new ListNode(2);
list.next.next.next = new ListNode(1);

palindromeLinkedList(list);

var threeSum = function (nums) {
  if (nums.length === 0) return [];
  const result = [];
  nums.sort((a, b) => a - b);

  for (let i = 0; i < nums.length - 2; i++) {
    //stop duplicates from occuring
    if (i > 0 && nums[i] === nums[i - 1]) continue;
    let left = i + 1;
    let right = nums.length - 1;

    while (left < right) {
      let sum = nums[left] + nums[right] + nums[i];
      if (sum === 0) {
        result.push([nums[left], nums[right], nums[i]]);
        //stop duplicates
        while (nums[left] == nums[left + 1]) left++;
        while (nums[right] == nums[right + 1]) right--;
        left++;
        right--;
      } else if (sum < 0) {
        left++;
      } else {
        right--;
      }
    }
  }

  return result;
};

threeSum([-1, 0, 1, 2, -1, -4]);

var maxArea = function (height) {
  let left = 0;
  let right = height.length - 1;
  let maxArea = 0;

  while (left < right) {
    let distance = right - left;
    let shorter = Math.min(height[left], height[right]);
    let area = distance * shorter;
    maxArea = Math.max(maxArea, area);

    if (height[left] < height[right]) {
      left++;
    } else {
      right--;
    }
  }

  return maxArea;
};

maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]);

var removeDuplicates = function (nums) {
  let index = 1;

  for (let i = 0; i < nums.length - 1; i++) {
    if (nums[i] != nums[i + 1]) {
      nums[index] = nums[i + 1];
      index++;
    }
  }

  return index;
};

removeDuplicates([1, 1, 2, 2, 3]);

var threeSumClosest = function (nums, target) {
  //sort the arrays to use two pointer technique
  nums.sort((a, b) => a - b);
  let diff = Number.MAX_SAFE_INTEGER;
  let result = 0;

  for (let i = 0; i < nums.length - 2; i++) {
    let left = i + 1;
    let right = nums.length - 1;

    while (left < right) {
      let sum = nums[i] + nums[left] + nums[right];

      if (sum === target) {
        return target;
      } else if (sum < target) {
        left++;
      } else {
        right--;
      }

      const difference = Math.abs(sum - target);

      if (difference < diff) {
        diff = difference;
        result = sum;
      }
    }
  }

  return result;
};

threeSumClosest([-1, 2, 1, -4], 1);

var fourSum = function (nums, target) {
  nums.sort((a, b) => a - b);

  let result = [];
  if (nums.length < 4) return [];

  for (let i = 0; i < nums.length - 3; i++) {
    for (let j = i + 1; j < nums.length - 2; j++) {
      let k = j + 1;
      let l = nums.length - 1;

      while (k < l) {
        let sum = nums[i] + nums[j] + nums[k] + nums[l];

        if (sum === target) {
          result.push([nums[i], nums[j], nums[k], nums[l]]);
          while (nums[k] === nums[k + 1]) k++;
          while (nums[l] === nums[l - 1]) l--;
          k++;
          l--;
        } else if (sum < target) {
          k++;
        } else {
          l--;
        }
      }

      while (nums[j] === nums[j + 1]) j++;
    }
    while (nums[i] === nums[i + 1]) i++;
  }

  return result;
};

fourSum([1, 0, -1, 0, -2, 2], 0);

var nextPermutation = function (nums) {
  if (nums.length <= 1) return;

  let leftHandSwap;

  // Loop through the provided numbers from right to left (excluding the first as we need something to compare it to)
  for (let i = nums.length - 2; i >= 0; i--) {
    // Check if this number is lower than the previous one (marks our left-hand swap)
    if (nums[i] < nums[i + 1]) {
      leftHandSwap = i;
      break;
    }
  }

  // Loop through the provided numbers from right to left
  for (let i = nums.length - 1; i > leftHandSwap; i--) {
    // If the number is bigger than the left-hand one
    if (nums[i] > nums[leftHandSwap]) {
      // Swap the numbers round
      [nums[i], nums[leftHandSwap]] = [nums[leftHandSwap], nums[i]];

      // Reverse the rest of the array
      let chopped = nums.splice(leftHandSwap + 1);
      chopped.sort((a, b) => a - b);
      nums.push(...chopped);
      return;
    }
  }

  // Right-hand swap not found, return lowest permutation instead
  nums.sort((a, b) => a - b);
};

nextPermutation([1, 2, 3]);

var isPalindrome = function (x) {
  let strX = x.toString();

  let left = 0;
  let right = strX.length - 1;

  while (left < right) {
    if (strX[left] !== strX[right]) {
      return false;
    }

    left++;
    right--;
  }

  return true;
};

isPalindrome(121);

var isValidPalindrome = function (s) {
  let sStr = s.toLowerCase();
  let filteredS = "";
  for (let i = 0; i < sStr.length; i++) {
    if (
      (sStr.charCodeAt(i) >= 65 && sStr.charCodeAt(i) <= 90) || // uppercase alphabets
      (sStr.charCodeAt(i) >= 97 && sStr.charCodeAt(i) <= 122) || // lowercase alphabets
      (sStr.charCodeAt(i) >= 48 && sStr.charCodeAt(i) <= 57) // digits
    ) {
      filteredS += sStr[i];
    }
  }

  let left = 0;
  let right = filteredS.length - 1;

  while (left < right) {
    if (filteredS[left] !== filteredS[right]) {
      return false;
    }

    left++;
    right--;
  }

  return true;
};

isValidPalindrome("a.");

var rotateRight = function (head, k) {
  if (head === null) return head;

  let len = 1;
  let tail = head;

  while (tail.next !== null) {
    tail = tail.next;
    len++;
  }

  tail.next = head;

  let count = len - (k % len);

  while (count > 0) {
    head = head.next;
    tail = tail.next;
    count--;
  }

  tail.next = null;

  return head;
};

const rotateListNode = new ListNode(1);
rotateListNode.next = new ListNode(2);
rotateListNode.next.next = new ListNode(3);
rotateListNode.next.next.next = new ListNode(4);
rotateListNode.next.next.next.next = new ListNode(5);

rotateRight(rotateListNode, 2);

function removeDuplicatesII(nums) {
  // let l = 0;
  // let r = 0;
  // while (r < nums.length) {
  //   let count = 1;
  //   while (r + 1 < nums.length && nums[r] === nums[r + 1]) {
  //     r++;
  //     count++;
  //   }
  //   for (let i = 0; i < Math.min(2, count); i++) {
  //     nums[l] = nums[r];
  //     l++;
  //   }
  //   r++;
  // }

  // return l;

  let left = 0;
  let right = 0;
  let count = 0;

  while (right < nums.length) {
    if (nums[right] !== nums[right - 1]) {
      count = 1;
      nums[left] = nums[right];
      left++;
    } else {
      count++;
      if (count <= 2) {
        nums[left] = nums[right];
        left++;
      }
    }
    right++;
  }

  return left; // Return the length of the modified array
}

removeDuplicatesII([1, 1, 2, 2, 2, 3]);

var deleteDuplicates = function (head) {
  let clone = head;
  let freqMap = {};

  while (clone != null) {
    if (!freqMap[clone.val]) {
      freqMap[clone.val] = 1;
    } else {
      freqMap[clone.val]++;
    }

    clone = clone.next;
  }

  let prev = new ListNode(0, head);
  clone = prev;

  while (clone != null) {
    //check over freqmap count and delete the node if it's greater than 1
    while (clone.next && freqMap[clone.next.val] > 1) {
      clone.next = clone.next.next;
    }
    clone = clone.next;
  }

  return prev.next;
};

const deleteDuplicatesHead = new ListNode(1);
deleteDuplicatesHead.next = new ListNode(1);
deleteDuplicatesHead.next.next = new ListNode(1);
deleteDuplicatesHead.next.next.next = new ListNode(2);
deleteDuplicatesHead.next.next.next.next = new ListNode(3);

deleteDuplicates(deleteDuplicatesHead);

var hasCycle = function (head) {
  if (!head) return false;

  let fast = head;
  let slow = head;

  while (fast != null) {
    if (!fast.next) {
      return false;
    } else {
      fast = fast.next.next;
      slow = slow.next;
    }

    if (fast === slow) return true;
  }

  return false;
};

const hasCycleHead = new ListNode(3);
hasCycleHead.next = new ListNode(2);
hasCycleHead.next.next = new ListNode(0);
hasCycleHead.next.next.next = new ListNode(4);
hasCycleHead.next.next.next.next = new ListNode(2);

hasCycle(hasCycleHead);

var lengthOfLargestSubstring = function (s) {
  let set = new Set();

  let left = 0;
  let right = 0;
  let longestSubstring = 0;

  while (right < s.length) {
    if (!set.has(s[right])) {
      set.add(s[right]);
      longestSubstring = Math.max(set.size, longestSubstring);
      right++;
    } else {
      set.delete(s[left]);
      left++;
    }
  }

  return longestSubstring;
};

lengthOfLargestSubstring("pwwkew");

var longestPalindrome = function (s) {
  let longest = "";
  function isPal(s, left, right) {
    while (left >= 0 && right <= s.length && s[left] == s[right]) {
      left--;
      right++;
    }

    return s.slice(left + 1, right);
  }

  for (let i = 0; i < s.length; i++) {
    let oddPal = isPal(s, i, i);
    let evenPal = isPal(s, i, i + 1);

    let longestPal = oddPal.length > evenPal.length ? oddPal : evenPal;

    if (longestPal.length > longest.length) {
      longest = longestPal;
    }
  }

  return longest;
};

longestPalindrome("babad");

var reverseList = function (head) {
  let prev = null;

  while (head) {
    let next = head.next;
    head.next = prev;
    prev = head;
    head = next;
  }

  return prev;
};

const reverseListNode = new ListNode(1);
reverseListNode.next = new ListNode(2);
reverseListNode.next.next = new ListNode(3);
reverseListNode.next.next.next = new ListNode(4);
reverseListNode.next.next.next.next = new ListNode(5);

reverseList(reverseListNode);

var preorderTraversal = function (root) {
  // Tree traversal in which root is visited first, then left and right node

  let arr = [];

  checkTree(root, arr);

  function checkTree(root, arr) {
    if (root === null) return root;

    arr.push(root.val);
    checkTree(root.left, arr);
    checkTree(root.right, arr);
  }

  return arr;
};

const inorderNode = new TreeNode(1);
inorderNode.left = null;
inorderNode.right = new TreeNode(2);
inorderNode.right.left = new TreeNode(3);

preorderTraversal(inorderNode);

let counter = 0;
//solving the problem in recursive ways. Learning recursion
var swapPairs = function (head) {
  //base case
  if (head === null || head.next === null) {
    return head;
  }

  //swap the first and second
  let second = head.next;
  head.next = second.next;
  second.next = head;

  //recursion for the remaining
  counter++;

  head.next = swapPairs(head.next);

  return second;
};

const swapPairsNode = new ListNode(1);
swapPairsNode.next = new ListNode(2);
swapPairsNode.next.next = new ListNode(3);
swapPairsNode.next.next.next = new ListNode(4);

swapPairs(swapPairsNode);

var myPow = function (x, n) {
  function calcPower(x, n) {
    if (x === 0) {
      return 0;
    }
    if (n === 0) {
      return 1;
    }

    let res = calcPower(x, Math.floor(n / 2));
    res = res * res;

    if (n % 2 == 1) {
      return res * x;
    }

    return res;
  }

  let ans = calcPower(x, Math.abs(n));

  if (n >= 0) {
    return ans;
  } else {
    return 1 / ans;
  }
};

myPow(2, 10);

var reorderList = function (head) {
  //find the mid
  let slow = head;
  let fast = head;

  while (fast.next && fast.next.next) {
    slow = slow.next;
    fast = fast.next.next;
  }

  let prev = null;

  //destroy the linked list from the mid
  let curr = slow.next;
  slow.next = prev;

  //reverse the linked list
  while (curr) {
    let temp = curr.next;
    curr.next = prev;
    prev = curr;
    curr = temp;
  }

  //merge the list
  let h1 = head;
  let h2 = prev;

  while (h2) {
    let temp = h1.next;
    h1.next = h2;
    h1 = h2;

    h2 = temp;
  }

  return head;
};

const reorderListNode = new ListNode(1);
reorderListNode.next = new ListNode(2);
reorderListNode.next.next = new ListNode(3);
reorderListNode.next.next.next = new ListNode(4);

reorderList(reorderListNode);

var diffWaysToCompute = function (input) {
  if (!isNaN(input)) {
    return [parseInt(input)];
  }

  let res = [];
  for (let i = 0; i < input.length; i++) {
    if (["-", "+", "*"].includes(input[i])) {
      let res1 = diffWaysToCompute(input.slice(0, i));
      let res2 = diffWaysToCompute(input.slice(i + 1));

      for (let j of res1) {
        for (let k of res2) {
          res.push(helper(j, k, input[i]));
        }
      }
    }
  }
  ``;
  return res;
};

var helper = function (m, n, op) {
  if (op === "+") {
    return m + n;
  } else if (op === "-") {
    return m - n;
  } else {
    return m * n;
  }
};

diffWaysToCompute("2-1-1");

var combinations = function (n, k) {
  let result = [];

  function dfs(index, current) {
    //base case
    if (current.length === k) {
      result.push([...current]);
    }

    for (let i = index; i <= n; i++) {
      current.push(i);
      //recurse
      dfs(i + 1, current);
      //backtrack
      current.pop();
    }
  }

  dfs(1, []);

  return result;
};

combinations(4, 2);

var permuteUnique = function (nums) {
  let result = [];
  let permutation = [];
  let count = {};

  // Count the frequency of each number
  for (let num of nums) {
    count[num] = (count[num] || 0) + 1;
  }

  // Recursive function to generate permutations
  var dfs = function () {
    // Base case: If the current permutation has the same length as the input array nums
    if (permutation.length === nums.length) {
      result.push([...permutation]);
      return;
    }

    // Iterate through each unique number
    for (let num of Object.keys(count)) {
      num = parseInt(num); // Convert key to number

      // Check if the current number's count is greater than 0
      if (count[num] > 0) {
        permutation.push(num); // Include the current number in the permutation
        count[num] -= 1; // Decrement the count

        dfs(); // Recursively generate permutations

        // Backtrack: Restore the count and remove the last added element from permutation
        count[num] += 1;
        permutation.pop();
      }
    }
  };

  dfs(); // Start the recursive process

  return result;
};

permuteUnique([1, 1, 2]);

var combinationSum2 = function (candidates, target) {
  let result = [];
  candidates.sort((a, b) => a - b);

  var dfs = function (target, start, comb) {
    if (target < 0) {
      return;
    }
    if (target === 0) {
      result.push([...comb]);
    }

    for (let i = start; i < candidates.length; i++) {
      if (i > start && candidates[i] === candidates[i - 1]) {
        continue;
      }
      if (candidates[i] > target) {
        break;
      }

      dfs(target - candidates[i], i + 1, [...comb, candidates[i]]);
    }
  };

  dfs(target, 0, []);
  return result;
};

combinationSum2([10, 1, 2, 7, 6, 1, 5], 8);

var combinationSum3 = function (k, n) {
  let result = [];

  var dfs = function (target, start, comb) {
    //base case
    if (target === 0 && comb.length === k) {
      result.push([...comb]);
      return;
    }

    for (let i = start; i <= 9; i++) {
      dfs(target - i, i + 1, [...comb, i]);
    }
  };

  dfs(n, 1, []);

  return result;
};

combinationSum3(3, 9);

var subsetsWithDup = function (nums) {
  let result = [[]];

  nums.sort((a, b) => a - b);

  var dfs = function (nums, res, currArr, start) {
    for (let i = start; i < nums.length; i++) {
      if (i === start || nums[i] !== nums[i - 1]) {
        currArr.push(nums[i]);
        res.push([...currArr]);
        dfs(nums, res, currArr, i + 1);
        currArr.pop();
      }
    }
  };

  dfs(nums, result, [], 0);

  return result;
};

subsetsWithDup([1, 2, 2]);

var maxProfit = function (prices) {
  let maxProfit = 0;
  let minPrice = prices[0];

  for (let i = 0; i < prices.length; i++) {
    minPrice = Math.min(minPrice, prices[i]);

    maxProfit += prices[i] - minPrice;

    minPrice = prices[i];
  }

  return maxProfit;
};

maxProfit([7, 1, 5, 3, 6, 4]);

var romanToInt = function (s) {
  const letter = {
    I: 1,
    V: 5,
    X: 10,
    L: 50,
    C: 100,
    D: 500,
    M: 1000,
  };
  let result = 0;

  for (let i = 0; i < s.length; i++) {
    if (i < s.length - 1 && letter[s[i]] < letter[s[i + 1]]) {
      result -= letter[s[i]];
    } else {
      result += letter[s[i]];
    }
  }

  return result;
};

romanToInt("IX");

var intToRoman = function (num) {
  const list = [
    "M",
    "CM",
    "D",
    "CD",
    "C",
    "XC",
    "L",
    "XL",
    "X",
    "IX",
    "V",
    "IV",
    "I",
  ];
  const valueList = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
  let result = "";

  while (num != 0) {
    for (let i = 0; i < list.length; i++) {
      if (num >= valueList[i]) {
        result += list[i];
        num -= valueList[i];
        break;
      }
    }
  }

  return result;
};

intToRoman(499);

var convert = function (s, numRows) {
  let count = 0;
  let direction = false;
  let result = new Array(numRows).fill("");

  for (let i = 0; i < s.length; i++) {
    let curr = s[i];

    result[count] += curr;

    if (count == 0 || count >= numRows - 1) direction = !direction;
    direction ? count++ : count--;
  }

  return result.join("");
};

convert("PaypalIsHiring", 3);

var reverseInteger = function (x) {
  let limit = Math.pow(2, 31) - 1;
  let rev = x.toString().split("").reverse().join("");
  let result = parseInt(rev);
  if (result > limit || result < -limit) {
    return 0;
  }

  if (x > 0) {
    return result;
  } else {
    return -result;
  }
};

reverseInteger(-123);

var tribonacci = function (n) {
  let t0 = 0;
  let t1 = 1;
  let t2 = 1;
  let sum = 0;

  for (let i = 3; i <= n; i++) {
    sum = t0 + t1 + t2;
    t0 = t1;
    t1 = t2;
    t2 = sum;
  }

  return sum;
};

tribonacci(25);

var minAvgDiff = function (nums) {
  let leftSum = 0;
  let rightSum = nums.reduce((a, b) => a + b);
  let minIndex = 0;
  let minDiff = rightSum;
  let n = nums.length;

  for (let i = 0; i < n; i++) {
    leftSum += nums[i];
    rightSum -= nums[i];

    const leftAverage = Math.floor(leftSum / (i + 1));
    const rightAverage = n - i - 1 === 0 ? 0 : Math.floor(rightSum / n - 1 - i);
    const currentDiff = Math.abs(leftAverage - rightAverage);

    if (currentDiff < minDiff) {
      minDiff = currentDiff;
      minIndex = i;
    }
  }

  return minIndex;
};

minAvgDiff([2, 5, 3, 9, 5, 3]);

var minimumRounds = function (tasks) {
  let map = new Map();
  let ans = 0;
  for (let task of tasks) {
    if (!map.has(task)) {
      map.set(task, 0);
    }
    map.set(task, map.get(task) + 1);
  }

  for (const [task, count] of map.entries()) {
    if (count == 1) return -1;
    ans += Math.floor((count + 2) / 3); // Perform integer division
  }

  return ans;
};

minimumRounds([2, 2, 3, 3, 2, 4, 4, 4, 4, 4]);

var pivotArray = function (nums, pivot) {
  let mid = [];
  let high = [];
  let low = [];

  for (let i = 0; i < nums.length; i++) {
    if (nums[i] > pivot) {
      high.push(nums[i]);
    } else if (nums[i] == pivot) {
      mid.push(nums[i]);
    } else {
      low.push(nums[i]);
    }
  }

  return low.concat(mid.concat(high));
};

pivotArray([9, 12, 5, 10, 14, 3, 10], 10);

var minimumSum = function (num) {
  num = num.toString().split("").sort().join("");

  let num1 = "";
  let num2 = "";

  for (let i = 0; i < num.length; i = i + 2) {
    num1 += num[i];
    num2 += num[i + 1];
  }

  let sum = 0;

  sum = +num1 + +num2;
  return sum;
};

minimumSum(2932);

var areNumbersAscending = function (s) {
  const words = s.split(" ");
  let prevNum = -Infinity;

  for (const word of words) {
    if (!isNaN(word)) {
      const current = +word;
      if (current <= prevNum) {
        return false;
      }
      prevNum = current;
    }
  }

  return true;
};

areNumbersAscending("1 box has 3 blue 4 red 6 green and 12 yellow marbles");

var buildArray = function (nums) {
  const result = [];

  for (let i = 0; i < nums.length; i++) {
    result.push(nums[nums[i]]);
  }

  return result;
};

buildArray([0, 2, 1, 5, 3, 4]);

var minOperations = function (nums) {
  if (nums.length <= 1) return 0;
  let increasedCount = 0;

  for (let i = 1; i < nums.length; i++) {
    if (nums[i] <= nums[i - 1]) {
      increasedCount = nums[i - 1] - nums[i] + 1;
      nums[i] = nums[i - 1] + 1;
    }
  }

  return increasedCount;
};

minOperations([1, 1, 1]);

var largestAltitude = function (gain) {
  let maxAltitude = 0;
  let currentAltitude = 0;

  for (let i = 0; i < gain.length; i++) {
    currentAltitude += gain[i];

    if (currentAltitude > maxAltitude) {
      maxAltitude = currentAltitude;
    }
  }

  return maxAltitude;
};

largestAltitude([-5, 1, 5, 0, -7]);

var minPartitions = function (n) {
  let max = "0";

  for (let c of n) {
    if (c > max) {
      max = c;
    }
  }

  return +max;
};

minPartitions("9");

var interpret = function (command) {
  let interpretedWord = "";
  const map = {
    G: "G",
    "()": "o",
    "(al)": "al",
  };
  let completeChar = "";
  for (let c of command) {
    completeChar += c;
    if (map[completeChar] != undefined) {
      interpretedWord += map[completeChar];
      completeChar = "";
    }
  }
  return interpretedWord;

  //One liner solution
  // return command.split("()").join("o").split("(al)").join("al")
};

interpret("G()()(al)");

var arrayStringsAreEqual = function (word1, word2) {
  let fullWord1 = "";
  let fullWord2 = "";

  for (let i = 0; i < word1.length; i++) {
    fullWord1 += word1[i];
  }

  for (let i = 0; i < word2.length; i++) {
    fullWord2 += word2[i];
  }

  return fullWord1 === fullWord2;

  //One liner
  //return word1.join("") === word2.join("");
};

arrayStringsAreEqual(["abc", "d", "defg"], ["abcddefg"]);

class Node {
  constructor(val, next = null, random = null) {
    this.val = val;
    this.next = next;
    this.random = random;
  }
}

function copyRandomList(head) {
  if (!head) return null;

  //Step 1: Clone nods and insert them next to original nodes
  let ptr = head;
  while (ptr) {
    const newNode = new Node(ptr.val);
    newNode.next = ptr.next;
    ptr.next = newNode;
    ptr = newNode.next;
  }

  //Step 2: Assign random pointers to cloned nodes
  ptr = head;
  while (ptr) {
    if (ptr.random) {
      ptr.next.random = ptr.random.next;
    }
    ptr = ptr.next.next;
  }

  //Step 3: Separate original and cloned linked lists
  const head_old = head;
  const head_new = head.next;
  let ptr_old = head_old;
  let ptr_new = head_new;

  while (ptr_old) {
    ptr_old.next = ptr_old.next.next;
    ptr_new.next = ptr_new.next ? ptr_new.next.next : null;
    ptr_old = ptr_old.next;
    ptr_new = ptr_new.next;
  }

  return head_new;
}

// Create nodes
const node01 = new Node(7);
const node2 = new Node(13);
const node3 = new Node(11);
const node4 = new Node(10);
const node5 = new Node(1);

// Link nodes
node01.next = node2;
node2.next = node3;
node3.next = node4;
node4.next = node5;

// Assign random pointers
node01.random = null; // Random pointer of node1 is null
node2.random = node1; // Random pointer of node2 points to node1
node3.random = node5; // Random pointer of node3 points to node5
node4.random = node3; // Random pointer of node4 points to node3
node5.random = node1; // Random pointer of node5 points to node1

copyRandomList(node1);

var removeElement = function (nums, val) {
  let index = 0;
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] !== val) {
      nums[index] = nums[i];
      index++;
    }
  }
  return index;
};

var reverse = function (arr, start, end) {
  while (start < end) {
    [arr[start], arr[end]] = [arr[end], arr[start]];
    start++;
    end--;
  }
};

var rotate = function (nums, k) {
  let n = nums.length;
  k %= n;

  //Reverse the entire array
  reverse(nums, 0, n - 1);
  //Reverse the first k elements
  reverse(nums, 0, k - 1);
  //Reverse the rest of the array
  reverse(nums, k, n - 1);

  return nums;
};

rotate([1, 2, 3, 4, 5, 6, 7], 3);

var canJumpII = function (n) {
  let totalJumps = 0;
  let length = n.length - 1;
  let coverage = 0;
  let lastJumPIdx = 0;

  if (nums.length === 1) return 0;

  for (let i = 0; i < nums.length; i++) {
    coverage = Math.max(coverage, i + nums[i]);

    if (i === lastJumPIdx) {
      lastJumPIdx = coverage;
      totalJumps++;

      if (coverage >= length) {
        return totalJumps;
      }
    }
  }

  return totalJumps;
};

canJumpII([2, 3, 1, 1, 4]);

var RandomizedSet = function () {
  this.map = new Map();
  this.list = [];
};

RandomizedSet.prototype.insert = function (val) {
  if (this.map.has(val)) {
    return false;
  }
  this.map.set(val, this.list.length);
  this.list.push(val);
  return true;
};

RandomizedSet.prototype.remove = function (val) {
  if (!this.map.has(val)) {
    return false;
  }
  const index = this.map.get(val);
  const lastElement = this.list[this.list.length - 1];

  // Swap the last element with the element to remove
  this.list[index] = lastElement;
  this.map.set(lastElement, index);

  // Remove the last element from the list
  this.list.pop();
  this.map.delete(val);

  return true;
};

/**
 * @return {number}
 */
RandomizedSet.prototype.getRandom = function () {
  const randomIndex = Math.floor(Math.random() * this.list.length);
  return this.list[randomIndex];
};

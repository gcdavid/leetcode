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
//     console.log(res[i], "res[i]", i, "index");
//     start2 = start2 * nums[i];
//     console.log(start2, "start2");
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
    // console.log(current, "current");
    if (previous[end] >= current[start]) {
      // console.log(previous[end] + ">=" + current[start]);

      //get the max value from the end of prev and current
      previous[end] = Math.max(previous[end], current[end]);
    } else {
      // console.log(current, "current");
      result.push(current);
      // console.log(previous + "=" + current);

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
      //backtract
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

levelOrder(node1);

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

console.log(uniquePaths(3, 7));

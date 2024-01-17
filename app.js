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

console.log(maxProduct([2, 3, -2, 4]));

// ============ 配置区域（请修改下面的 GitHub 信息） ============
const githubUsername = 'owencalstroy'; // 修改为你的 GitHub 用户名
const repoName = 'CSnAI';             // 修改为你的仓库名
const branch = 'main';                          // 修改为你的分支名（例如 main 或 master）

// ============ 定义左侧文档目录结构 ============
// 请根据需要修改下面的结构，key 为文件夹/文件名称，
// 如果值为对象表示存在下一级目录，
// 如果值为字符串，则表示一个 Markdown 文件，字符串为自定义显示名称。
// 此结构支持无限级嵌套。
const docStructure = {
  "C++ Books": {
    "C++ STL": {
      "ch2.md": "Chap 2: C++ 及其标准程序库简介",
      "ch3.md": "Chap 3: 一般概念"
    },
  },
  "ml": {
    "book1": {
      "ch1.md": "ML Book1 - 第一章",
      "ch2.md": "ML Book1 - 第二章"
    },
    "book2": {
      "ch1.md": "ML Book2 - 第一章",
      "ch2.md": "ML Book2 - 第二章"
    }
  },
  "llm": {
    "book1": {
      "ch1.md": "LLM Book1 - 第一章",
      "ch2.md": "LLM Book1 - 第二章"
    },
    "book2": {
      "ch1.md": "LLM Book2 - 第一章",
      "ch2.md": "LLM Book2 - 第二章"
    }
  }
};

// ============ 搜索相关变量初始化 ============
let lunrIndex;
let searchData = [];

/**
 * 递归构建菜单项
 * @param {Object} obj - 当前层级的目录结构对象
 * @param {string} parentPath - 上一级的路径（用于构造完整的文件路径）
 * @returns {HTMLElement} 构建好的 <ul> 元素
 */
function buildMenuItems(obj, parentPath) {
  const ul = document.createElement('ul');
  ul.className = 'nav flex-column';
  for (const key in obj) {
    const li = document.createElement('li');
    li.className = 'nav-item';
    // 如果值是对象，则表示为文件夹（可展开/收起）
    if (typeof obj[key] === 'object') {
      // 创建文件夹链接（包含箭头图标）
      const a = document.createElement('a');
      a.className = 'nav-link folder-link';
      a.href = '#';
      // 创建箭头 span，初始为向右
      const arrow = document.createElement('span');
      arrow.className = 'toggle-arrow';
      arrow.innerHTML = '&#9654; '; // 向右箭头
      a.appendChild(arrow);
      // 显示文件夹名称（直接用 key，可根据需要修改为自定义名称）
      const span = document.createElement('span');
      span.textContent = key;
      a.appendChild(span);
      // 点击文件夹链接切换子菜单显示状态
      a.addEventListener('click', function(e) {
        e.preventDefault();
        if (childUl.style.display === 'none' || childUl.style.display === '') {
          childUl.style.display = 'block';
          arrow.innerHTML = '&#9660; '; // 向下箭头
        } else {
          childUl.style.display = 'none';
          arrow.innerHTML = '&#9654; '; // 向右箭头
        }
      });
      li.appendChild(a);
      // 更新路径：如果 parentPath 为空，则直接用 key，否则拼接 parentPath/key
      const newParentPath = parentPath ? (parentPath + '/' + key) : key;
      // 递归构建子菜单
      const childUl = buildMenuItems(obj[key], newParentPath);
      childUl.style.display = 'none'; // 默认折叠状态
      li.appendChild(childUl);
    } else {
      // 如果值不是对象，则表示为文件（Markdown 文件），key 为文件名，值为自定义显示名称
      const a = document.createElement('a');
      a.className = 'nav-link';
      a.href = '#';
      a.textContent = obj[key]; // 显示自定义的文件标题
      // 构造文件完整路径，parentPath 需传入当前层级路径（例如 "doc/cpp/book1"）
      const filePath = parentPath ? (parentPath + '/' + key) : key;
      a.dataset.path = filePath; 
      a.addEventListener('click', function(e) {
        e.preventDefault();
        loadMarkdown(this.dataset.path);
      });
      li.appendChild(a);
    }
    ul.appendChild(li);
  }
  return ul;
}

/**
 * 构建左侧边栏目录（调用递归函数）
 * 初始路径为 "doc"（即 Markdown 文件存放的根目录下的 doc 文件夹）
 */
function buildDocList() {
  const docNav = document.getElementById('docNav');
  docNav.innerHTML = '';
  const menu = buildMenuItems(docStructure, 'doc'); // 若 Markdown 文件存放在 "doc" 文件夹下，请保持此处不变
  docNav.appendChild(menu);
}

/**
 * 根据文件路径加载 Markdown 文件（通过 GitHub Raw URL）
 * @param {string} path - 文件相对于仓库根目录的路径
 */
function loadMarkdown(path) {
  const url = `https://raw.githubusercontent.com/${githubUsername}/${repoName}/${branch}/${path}`;
  fetch(url)
    .then(response => response.text())
    .then(text => {
      // 使用 marked.js 渲染 Markdown 内容
      const html = marked(text);
      document.getElementById('markdownContent').innerHTML = html;
      generateTOC(text);
    })
    .catch(error => {
      console.error('加载 Markdown 失败:', error);
      document.getElementById('markdownContent').innerHTML = '<p>加载文档失败。</p>';
    });
}

/**
 * 根据 Markdown 文本生成右侧目录（TOC），提取 ## 至 ###### 级别标题
 */
function generateTOC(markdownText) {
  const lines = markdownText.split('\n');
  const toc = document.getElementById('toc');
  toc.innerHTML = '';
  lines.forEach(line => {
    const headerMatch = line.match(/^(#{2,6})\s+(.*)/);
    if (headerMatch) {
      const level = headerMatch[1].length; // 标题级别
      const title = headerMatch[2];
      const li = document.createElement('li');
      li.className = 'nav-item';
      li.textContent = title;
      li.style.marginLeft = (level - 2) * 15 + 'px';
      toc.appendChild(li);
    }
  });
}

/**
 * 加载首页或关于页面
 */
function loadPage(page) {
  const path = page === 'home' ? 'home.md' : 'about.md';
  loadMarkdown(path);
}

/**
 * 初始化搜索索引（从 search_index.json 加载数据）
 */
function initSearch() {
  fetch('search_index.json')
    .then(response => response.json())
    .then(data => {
      searchData = data;
      lunrIndex = lunr(function () {
        this.ref('id');
        this.field('title');
        this.field('content');
        data.forEach(doc => this.add(doc));
      });
    })
    .catch(error => console.error('加载搜索索引失败:', error));
}

/**
 * 执行搜索并显示结果
 */
function performSearch(query) {
  const results = lunrIndex.search(query);
  const resultsContainer = document.getElementById('searchResults');
  resultsContainer.innerHTML = '';
  if (results.length > 0) {
    results.forEach(result => {
      const doc = searchData.find(d => d.id === result.ref);
      const resultItem = document.createElement('div');
      resultItem.innerHTML = `<a href="#" data-path="${doc.url}">${doc.title}</a>`;
      resultItem.querySelector('a').addEventListener('click', function(e) {
        e.preventDefault();
        loadMarkdown(this.dataset.path);
        resultsContainer.style.display = 'none';
      });
      resultsContainer.appendChild(resultItem);
    });
    resultsContainer.style.display = 'block';
  } else {
    resultsContainer.innerHTML = '<p>没有找到相关结果。</p>';
    resultsContainer.style.display = 'block';
  }
}

// ============ 事件绑定 ============
// 顶部导航按钮事件
document.getElementById('homeBtn').addEventListener('click', function(e) {
  e.preventDefault();
  loadPage('home');
});
document.getElementById('aboutBtn').addEventListener('click', function(e) {
  e.preventDefault();
  loadPage('about');
});
// 搜索按钮事件
document.getElementById('searchBtn').addEventListener('click', function() {
  const query = document.getElementById('searchInput').value;
  if (query) {
    performSearch(query);
  }
});

// ============ 页面初始化 ============
// 构建左侧目录、初始化搜索、默认加载首页
buildDocList();
initSearch();
loadPage('home'); // 默认加载首页

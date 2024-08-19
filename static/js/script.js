const promptInput = document.getElementById("userInput");
const promptInput2 = document.getElementById("userInput2");
const chatContainer = document.getElementById("chatContainer");
const secondaryChatContainer = document.getElementById("secondaryChatContainer");
const typingIndicator = document.getElementById("typingIndicator");
const typingIndicator2 = document.getElementById("typingIndicator2");
const sidebar = document.getElementById("sidebar");
const sidebarContent = document.getElementById("sidebarContent");
const secondarySidebarContent = document.getElementById("secondarySidebarContent");
const imageContainer = document.getElementById("imageContainer");

async function sendMessage() {
  const prompt = promptInput.value.trim();
  if (!prompt && imageContainer.children.length === 0) {
    alert("Please enter a message or add an image.");  // Browser pop up message
    return;
  }

  // Collect image data
  const images = Array.from(imageContainer.querySelectorAll('.img-preview'))
    .map(img => img.src.split(',')[1]); // Extract base64 data

  addMessage(prompt, 'user', images);
  promptInput.value = "";

  showTypingIndicator();

  const generatedText = await generateText(prompt, images);
  addMessage(generatedText, 'bot');

  hideTypingIndicator();
  //clearImagePreviews(); Add this code if you want the image in the imageContainer disappear if the user sends the image.
}



async function sendMessage2() {
  const prompt2 = promptInput2.value.trim();
  if (!prompt2 && imageContainer.children.length === 0) {
    alert("Please enter a message or add an image.");  // Browser pop up message
    return;
  }

  // Collect image data
  const images = Array.from(imageContainer.querySelectorAll('.img-preview'))
    .map(img => img.src.split(',')[1]); // Extract base64 data

  addMessage2(prompt2, 'user', images);
  promptInput2.value = "";

  showTypingIndicator2();

  const generatedText2 = await generateText2(prompt2, images);
  addMessage2(generatedText2, 'bot');

  hideTypingIndicator2();
  //clearImagePreviews(); Add this code if you want the image in the imageContainer disappear if the user sends the image.
}


async function generateText(prompt, images) {
  try {
    const response = await fetch("http://127.0.0.1:8080/generate_text_stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt, images }),
    });

    if (!response.ok) {
      console.error("Error:", response.statusText);
      return "Error occurred while generating response.";
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let isFinished = false;
    let generatedTextContent = "";

    while (!isFinished) {
      const { done, value } = await reader.read();
      if (done) {
        isFinished = true;
        break;
      }
      generatedTextContent += decoder.decode(value, { stream: true });
    }

    return generatedTextContent;
  } catch (error) {
    console.error("Error:", error);
    return "An error occurred.";
  }
}




async function generateText2(prompt2, images) {
  try {
    const response2 = await fetch("http://127.0.0.1:8080/generate_SEC_10K_text_stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt2, images }),
    });

    if (!response2.ok) {
      console.error("Error:", response2.statusText);
      return "Error occurred while generating response.";
    }

    const reader2 = response2.body.getReader();
    const decoder2 = new TextDecoder();
    let isFinished2 = false;
    let generatedTextContent2 = "";

    while (!isFinished2) {
      const { done, value } = await reader2.read();
      if (done) {
        isFinished2 = true;
        break;
      }
      generatedTextContent2 += decoder2.decode(value, { stream: true });
    }

    return generatedTextContent2;
  } catch (error) {
    console.error("Error:", error);
    return "An error occurred.";
  }
}




function addMessage(text, type, images = []) {

  let md = window.markdownit();

  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${type}`;

  const messageContent = document.createElement("div");
  messageContent.className = "message-bubble fadeIn";
  const render = () => {
    messageContent.innerHTML = md.render(text);
  }
  render()

  images.forEach(src => {
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${src}`;
    img.classList.add("message-image");
    messageContent.appendChild(img);
  });

  messageDiv.appendChild(messageContent);
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addMessage2(text, type, images = []) {

  let md = window.markdownit();

  const messageDiv2 = document.createElement("div");
  messageDiv2.className = `message ${type}`;

  const messageContent2 = document.createElement("div");
  messageContent2.className = "message-bubble fadeIn";
  const render = () => {
    messageContent2.innerHTML = md.render(text);
  }
  render()

  images.forEach(src => {
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${src}`;
    img.classList.add("message-image");
    messageContent2.appendChild(img);
  });

  messageDiv2.appendChild(messageContent2);
  secondaryChatContainer.appendChild(messageDiv2);
  secondaryChatContainer.scrollTop = secondaryChatContainer.scrollHeight;
}

function clearImagePreviews() {
  while (imageContainer.firstChild) {
    imageContainer.removeChild(imageContainer.firstChild);
  }
  checkImageContainerVisibility();
}

let typingTimeout;

function showTypingIndicator() {
  clearTimeout(typingTimeout);
  typingIndicator.style.display = "inline-block";
}

function hideTypingIndicator() {
  typingTimeout = setTimeout(() => {
    typingIndicator.style.display = "none";
  }, 1000);
}


let typingTimeout2;

function showTypingIndicator2() {
  clearTimeout(typingTimeout2);
  typingIndicator2.style.display = "inline-block";
}

function hideTypingIndicator2() {
  typingTimeout2 = setTimeout(() => {
    typingIndicator2.style.display = "none";
  }, 1000);
}


function handleKeyPress(event) {
  if (event.key === "Enter") {
    sendMessage();
  }
}

function secondaryHandleKeyPress(event) {
  if (event.key === "Enter") {
    sendMessage2();
  }
}



function toggleSidebar() {
  if (sidebar.style.width === "500px") {
    sidebar.style.width = "0";
    sidebarContent.style.display = "none";
  } else {
    sidebar.style.width = "500px";
    sidebarContent.style.display = "block";
  }
}

function secondaryToggleSidebar() {
  if (sidebar.style.width === "500px") {
    sidebar.style.width = "0";
    secondarySidebarContent.style.display = "none";
  } else {
    sidebar.style.width = "500px";
    secondarySidebarContent.style.display = "block";
    addMessage2("Hello, please ask me the SEC data you would like to know about!", 'bot');
    addMessage2("For this demo, only AAPL and TSLA SEC data is available.", 'bot');
  }
}

window.onload = () => addMessage("Hello! How can I assist you today?", 'bot');

document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('userInput');
  const textInput2 = document.getElementById('userInput2');

  textInput.addEventListener('paste', (event) => {
    const items = (event.clipboardData || window.clipboardData).items;
    for (const item of items) {
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();
        const reader = new FileReader();
        reader.onload = (event) => {
          displayImage(event.target.result);
        };
        reader.readAsDataURL(file);
        event.preventDefault();
      }
    }
  });

  textInput2.addEventListener('paste', (event) => {
    const items = (event.clipboardData || window.clipboardData).items;
    for (const item of items) {
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();
        const reader = new FileReader();
        reader.onload = (event) => {
          displayImage(event.target.result);
        };
        reader.readAsDataURL(file);
        event.preventDefault();
      }
    }
  });

  function displayImage(src) {
    const imgContainer = document.createElement('div');
    imgContainer.classList.add('img-preview-container');

    const img = document.createElement('img');
    img.src = src;
    img.classList.add('img-preview');

    const removeButton = document.createElement('button');
    removeButton.classList.add('remove-button');
    removeButton.textContent = '✖';
    removeButton.addEventListener('click', () => {
      imgContainer.remove();
      checkImageContainerVisibility();
    });

    imgContainer.appendChild(img);
    imgContainer.appendChild(removeButton);
    imageContainer.appendChild(imgContainer);
    checkImageContainerVisibility();

    const all_images = imageContainer.querySelectorAll('.img-preview-container');
    all_images.forEach(img => img.style.width = `${100 / all_images.length - 10}%`);
  }

  function checkImageContainerVisibility() {
    if (imageContainer.children.length > 0) {
      imageContainer.classList.remove('hidden');
    } else {
      imageContainer.classList.add('hidden');
    }
  }

  // Initial check to hide image container if empty
  checkImageContainerVisibility();
});















// get_tricker
document.getElementById('getTickerButton').addEventListener('click', async function() {
  const button = this;
  showLoading(button);
  
  try {
    await getTicker();
  } catch (error) {
    console.error('Error:', error);
  } finally {
    hideLoading(button);
  }
});

async function getTicker() {
  const companyName = new Array(
    document.getElementById('companyName1').value,
    document.getElementById('companyName2').value,
    document.getElementById('companyName3').value,
    document.getElementById('companyName4').value
  )
  const announcementDiv = document.getElementById('announcement');
  const resultDiv = new Array(
    document.getElementById('result1'),
    document.getElementById('result2'),
    document.getElementById('result3'),
    document.getElementById('result4')
  )

  const tickerInput = new Array(
    document.getElementById('ticker1'),
    document.getElementById('ticker2'),
    document.getElementById('ticker3'),
    document.getElementById('ticker4')
  )
  
  // Clear previous results
  for(let i = 0; i<4; i++) {
    resultDiv[i].innerHTML = '';
  }

  // empty input 처리
  let inputCnt = 0;
  for(let i = 0; i<4; i++) {
    if(companyName[i]) inputCnt++;
  }
  if(inputCnt<2) {
    announcementDiv.innerHTML = 'Please enter at least two correct company name.';
    alert('Please enter at least two correct company name.')
    return false;
  }

  for(let i = 0; i < 4; i++) {
    if(companyName[i] === '') continue;

    try {
      const response = await fetch(`/get_ticker?company_name=${encodeURIComponent(companyName[i])}`);

      if (!response.ok) {
        resultDiv[i].innerHTML = 'Network response was not ok';
        continue;
      }

      const data = await response.json();
      if (!data.success) {
        resultDiv[i].innerHTML = data.message;
      } else {
        resultDiv[i].innerHTML = `The ticker for '${companyName[i]}' is: ${data.ticker}`;
        tickerInput[i].value = data.ticker;
      }
    } catch (error) {
      resultDiv[i].innerHTML = 'An error occurred while fetching the ticker';
      console.error('Error:', error);
    }
  }
}

function checkValidStartYearInput(startyear) {
  // YYYY-MM-DD 형식을 검사하는 정규 표현식
  const regex = /^\d{4}-\d{2}-\d{2}$/;

  // 정규 표현식과 매칭되지 않으면 false 반환
  if (!startyear.match(regex)) {
      return false;
  }

  // 날짜 유효성 검사 (월은 1-12, 일은 해당 월에 맞는 값이어야 함)
  const date = new Date(startyear);
  const [year, month, day] = startyear.split('-').map(Number);

  // 년도와 월의 비교를 통해 유효한 날짜인지 확인
  if (date.getFullYear() === year && (date.getMonth() + 1) === month && date.getDate() === day) {
      return true;
  } else {
      return false;
  }
}

document.getElementById('backtestResultButton').addEventListener('click', async function() {
  const button = this;
  showLoading(button);

  const cs_model = document.getElementById('cs_model').value;
  const ts_model = document.getElementById('ts_model').value;
  const tickers = [
    document.getElementById('ticker1').value,
    document.getElementById('ticker2').value,
    document.getElementById('ticker3').value,
    document.getElementById('ticker4').value
  ];
  
  const startyear = document.getElementById('startyear').value;
  const announcementDiv = document.getElementById('announcement');
  const filteredTickers = tickers.filter(ticker => ticker !== '');

  // get ticker 버튼이 아직 눌리지 않은 경우
  if(filteredTickers.length < 2) {
    announcementDiv.innerHTML = "Please enter at least two correct company name. Then click 'Get Ticker' button first.";
    alert("Please input at least two correct company name. Then click 'Get Ticker' button first.");
    hideLoading(button);
    return;
  }

  // startyear가 입력되지 않은 경우
  if(startyear == '') {
    alert("Please input start year.");
    hideLoading(button);
    return;
  }      
  
  // startyear가 valid한지 확인
  if(!checkValidStartYearInput(startyear)) {
    alert("Invalid start year. Please follow the format 'YYYY-MM-DD'.");
    hideLoading(button);
    return;
  }

  console.log('cs_model:', cs_model);
  console.log('ts_model:', ts_model);
  console.log('tickers:', tickers);
  console.log('startyear:', startyear);
  console.log(filteredTickers);

  try {
    const response = await fetch('/Backtest_result', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cs_model, ts_model, tickers: filteredTickers, startyear }),
    });

    if (!response.ok) {
      console.error('Error:', response.statusText);
      hideLoading(button);
      return;
    }

    const data = await response.json();
    if (data.error) {
      console.error('Error:', data.error);
      hideLoading(button);
      return;
    }
    document.getElementById('backtestResult').innerHTML = `
      <div class="image" id="image1">
        <img src="data:image/png;base64,${data.port_weights_img}" alt="Portfolio Weights">
      </div>
      <div class="image" id="image2">
        <img src="data:image/png;base64,${data.asset_performance_img}" alt="Asset Performance">
      </div>
      <div class="image" id="image3">
        <img src="data:image/png;base64,${data.portfolio_performance_img}" alt="Portfolio Performance">
      </div>
    `;
    document.getElementById('backtestResult').classList.add('active');
    document.getElementById('detailedReport').classList.remove('active');
  } catch (error) {
    console.error('Error:', error);
  } finally {
    hideLoading(button);
  }
});


document.getElementById('detailedReportButton').addEventListener('click', async function() {
  const button = this;
  showLoading(button);

  const cs_model = document.getElementById('cs_model').value;
  const ts_model = document.getElementById('ts_model').value;
  const tickers = [
    document.getElementById('ticker1').value,
    document.getElementById('ticker2').value,
    document.getElementById('ticker3').value,
    document.getElementById('ticker4').value
  ];

  const startyear = document.getElementById('startyear').value;
  const announcementDiv = document.getElementById('announcement');
  const filteredTickers = tickers.filter(ticker => ticker !== '');

  // get ticker 버튼이 아직 눌리지 않은 경우
  if(filteredTickers.length < 2) {
    announcementDiv.innerHTML = "Please enter at least two correct company name. Then click 'Get Ticker' button first.";
    alert("Please input at least two correct company name. Then click 'Get Ticker' button first.");
    hideLoading(button);
    return;
  }
  // startyear가 입력되지 않은 경우
  if(startyear == '') {
    alert("Please input start year.");
    hideLoading(button);
    return;
  }  
  
  // startyear가 valid한지 확인
  if(!checkValidStartYearInput(startyear)) {
    alert("Invalid start year. Please follow the format 'YYYY-MM-DD'.");
    hideLoading(button);
    return;
  }

  try {
    const response = await fetch('/generate_html_report', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cs_model, ts_model, tickers: filteredTickers, startyear }),
    });

    if (!response.ok) {
      console.error('Error:', response.statusText);
      hideLoading(button);
      return;
    }

    const data = await response.json();
    if (data.error) {
      console.error('Error:', data.error);
      hideLoading(button);
      return;
    }

    document.getElementById('detailedReport').innerHTML = data.report_html;
    document.getElementById('detailedReport').classList.add('active');
    document.getElementById('backtestResult').classList.remove('active');
  } catch (error) {
    console.error('Error:', error);
  } finally {
    hideLoading(button);
  }
});

function showLoading(button) {
  button.disabled = true;
  const spinner = document.createElement('div');
  spinner.className = 'spinner';
  button.appendChild(spinner);
}

function hideLoading(button) {
  button.disabled = false;
  const spinner = button.querySelector('.spinner');
  if (spinner) {
    button.removeChild(spinner);
  }
}
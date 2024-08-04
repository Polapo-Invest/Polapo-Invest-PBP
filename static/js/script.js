const promptInput = document.getElementById("userInput");
const chatContainer = document.getElementById("chatContainer");
const typingIndicator = document.getElementById("typingIndicator");
const sidebar = document.getElementById("sidebar");
const sidebarContent = document.getElementById("sidebarContent");
const imageContainer = document.getElementById("imageContainer");
console.log("This is a test message.");

// console.log = function(message) {
//   // 예제: 기본 동작을 유지하면서 추가 동작 수행
//   alert(message); // 메시지를 알림으로도 표시
//   console.warn(message); // 경고로 출력
// };

console.log("This is a test message.");

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

async function generateText(prompt, images) {
  try {
    const response = await fetch("https://opt-wep-mi7kcwnijq-uc.a.run.app/generate_text_stream", {
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
    const img = document.createElement("chat-img");
    img.src = `data:image/png;base64,${src}`;
    img.classList.add("message-image");
    messageContent.appendChild(img);
  });

  messageDiv.appendChild(messageContent);
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
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

function handleKeyPress(event) {
  if (event.key === "Enter") {
    sendMessage();
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

window.onload = () => addMessage("Hello! How can I assist you today?", 'bot');

document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('userInput');

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

  function displayImage(src) {
    const imgContainer = document.createElement('div');
    imgContainer.classList.add('img-preview-container');

    const img = document.createElement('chat-img');
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
async function getTicker() {
  const button = document.getElementById('getTickerButton');
  showLoading(button);

  const companyName = new Array(
    document.getElementById('companyName1').value,
    document.getElementById('companyName2').value,
    document.getElementById('companyName3').value,
    document.getElementById('companyName4').value
  )

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
  let emptyflag = false;
  for(let i = 0; i<4; i++) {
    if(!companyName[i]) {
      resultDiv[i].innerHTML = 'Please enter a company name.';
      emptyflag = true;
    }
  }
  if(emptyflag) {
    hideLoading(button);
    return false;
  }
  

  try {
    for(let i=0; i<4; i++) {
      const response = await fetch(`/get_ticker?company_name=${encodeURIComponent(companyName[i])}`);
      
      if (!response.ok) {
        resultDiv[i].innerHTML = 'Network response was not ok';
      } else {
        const data = await response.json();
        if(!data.success) {
          resultDiv[i].innerHTML = data.message;
        } else {
          resultDiv[i].innerHTML = `The ticker for '${companyName[i]}' is: ${data.ticker}`;
          tickerInput[i].value = data.ticker
        }
      }
    }
  } catch (error) {
    console.error('Error:', error);
    for(let i = 0; i<4; i++) {
      resultDiv[i].innerHTML = 'An error occurred while fetching the ticker.';
    }
  } finally {
    hideLoading(button);
}
}

function checkStartYear() {
  const inputField = document.getElementById('startyear');
  const approvalMessage = document.getElementById('approvalMessage');
  const disapprovalMessage = document.getElementById('disapprovalMessage');
  const correctValue = 'apple';

  if (inputField.value === correctValue) {
      approvalMessage.style.display = 'inline';
      disapprovalMessage.style.display = 'none';
  } else {
      approvalMessage.style.display = 'none';
      disapprovalMessage.style.display = 'inline';
  }
}

// Backtest Result and Detailed Report buttons
document.getElementById('backtestResultButton').addEventListener('click', function() {
  const button = this;
  showLoading(button);
  const cs_model = document.getElementById('cs_model').value;
  const ts_model = document.getElementById('ts_model').value;
  const tickers = new Array(
    document.getElementById('ticker1').value,
    document.getElementById('ticker2').value,
    document.getElementById('ticker3').value,
    document.getElementById('ticker4').value
  )
  const startyear = document.getElementById('startyear').value;
  
  console.log('cs_model:', cs_model);
  console.log('ts_model:', ts_model);
  console.log('tickers:', tickers);
  console.log('startyear:', startyear);

  fetch('/Backtest_result', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ cs_model: cs_model, ts_model: ts_model, tickers: tickers, startyear: startyear }),
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        console.error('Error:', data.error);
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
    })
    .catch(error => {
      console.error('Error:', error);
    });
  })
  .catch(error => {
    console.error('Error:', error);
    hideLoading(button);
  });

document.getElementById('detailedReportButton').addEventListener('click', function() {
  const button = this;
  showLoading(button);
  const cs_model = document.getElementById('cs_model').value;
  const ts_model = document.getElementById('ts_model').value;
  const tickers = new Array(
    document.getElementById('ticker1').value,
    document.getElementById('ticker2').value,
    document.getElementById('ticker3').value,
    document.getElementById('ticker4').value
  )
  const startyear = document.getElementById('startyear').value;

  fetch('/generate_html_report', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ cs_model: cs_model, ts_model: ts_model, tickers: tickers, startyear: startyear }),
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        console.error('Error:', data.error);
        return;
      }
      document.getElementById('detailedReport').innerHTML = data.report_html;
      document.getElementById('detailedReport').classList.add('active');
      document.getElementById('backtestResult').classList.remove('active');
    })
    .catch(error => {
      console.error('Error:', error);
    })
    .finally(() => {
      hideLoading(button);
    });
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
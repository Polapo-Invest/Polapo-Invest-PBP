const promptInput = document.getElementById("userInput");
const chatContainer = document.getElementById("chatContainer");
const typingIndicator = document.getElementById("typingIndicator");
const sidebar = document.getElementById("sidebar");
const sidebarContent = document.getElementById("sidebarContent");
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

function addMessage(text, type, images = []) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${type}`;

  const messageContent = document.createElement("div");
  messageContent.className = "message-bubble fadeIn";
  messageContent.innerHTML = `<p>${text}</p>`;

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

// Backtest Result and Detailed Report buttons
document.getElementById('backtestResultButton').addEventListener('click', function() {
  const cs_model = document.getElementById('cs_model').value;
  const ts_model = document.getElementById('ts_model').value;

  fetch('/Backtest_result', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ cs_model: cs_model, ts_model: ts_model }),
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        console.error('Error:', data.error);
        return;
      }

      function loadImage(src, alt) {
        return new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = reject;
          img.src = src;
          img.alt = alt;
        });
      }

      Promise.all([
        loadImage(`data:image/png;base64,${data.port_weights_img}`, "Portfolio Weights"),
        loadImage(`data:image/png;base64,${data.asset_performance_img}`, "Asset Performance"),
        loadImage(`data:image/png;base64,${data.portfolio_performance_img}`, "Portfolio Performance")
      ]).then(images => {
        const backtestResult = document.getElementById('backtestResult');
        backtestResult.innerHTML = '';
        images.forEach((img, index) => {
          const container = document.createElement('div');
          container.className = 'image-container';
          container.id = `image${index + 1}`;
          container.appendChild(img);
          backtestResult.appendChild(container);
        });
        backtestResult.classList.add('active');
        document.getElementById('detailedReport').classList.remove('active');
      }).catch(error => {
        console.error('Error loading images:', error);
      });
    })
    .catch(error => {
      console.error('Error:', error);
    });
});

document.getElementById('detailedReportButton').addEventListener('click', function() {
  const cs_model = document.getElementById('cs_model').value;
  const ts_model = document.getElementById('ts_model').value;

  fetch('/generate_html_report', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ cs_model: cs_model, ts_model: ts_model }),
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
    });
});

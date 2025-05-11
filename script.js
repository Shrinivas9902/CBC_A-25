// Handle text input and emotion detection
document.getElementById("analyze_button").addEventListener("click", function() {
    const userInput = document.getElementById("user_input").value.trim();
  
    if (userInput === "") {
      alert("Please enter your feelings.");
      return;
    }
  
    // Analyze the emotion (this would be done using a machine learning model in a real app)
    let emotion = "neutral";
    let copingStrategies = [];
  
    // Simple analysis based on keywords (for demonstration purposes)
    if (userInput.includes("happy") || userInput.includes("joy")) {
      emotion = "positive";
      copingStrategies = [
        "Keep up the positive thoughts!",
        "Practice gratitude daily.",
        "Stay connected with loved ones."
      ];
    } else if (userInput.includes("sad") || userInput.includes("down")) {
      emotion = "negative";
      copingStrategies = [
        "Take deep breaths.",
        "Reach out to a friend or therapist.",
        "Write down your thoughts."
      ];
    } else {
      emotion = "neutral";
      copingStrategies = [
        "Try mindfulness meditation.",
        "Keep a journal.",
        "Take short walks to clear your mind."
      ];
    }
  
    // Display the result
    document.getElementById("emotion").textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    document.getElementById("coping_strategies").innerHTML = copingStrategies.map(item => `<li>${item}</li>`).join("");
    
    document.getElementById("emotion_result").classList.remove("hidden");
  });
  
/**
 * Krishi Mitra (ಕೃಷಿ ಮಿತ್ರ / कृषि मित्र) - Client Controller
 * Complete Agricultural AI Assistant with:
 * - Speech-to-Text Voice Input (Microphone in EN, KN, HI)
 * - Text-to-Speech Voice Output (Audio Readout in EN, KN, HI)
 * - Grad-CAM Attention Heatmap Visualization (Wide Layout)
 * - 16L Knapsack Backpack Sprayer & Land Acreage Dosage Calculator
 * - Real-world Weather Spraying Advisory & WhatsApp Sharing
 * - Zero loops, clean asynchronous event architecture
 */

document.addEventListener("DOMContentLoaded", function () {

    // =========================================================
    // 1. TRILINGUAL DICTIONARY (EN, KN, HI)
    // =========================================================
    const translations = {
        en: {
            tagline: "Smart Crop Doctor & Precision Farm Assistant",
            nav_home: "🏠 Home",
            nav_scan: "🔍 Crop Doctor",
            nav_disease: "📖 Disease Info",
            nav_care: "🌱 Care Tips",
            nav_yield: "📊 Yield Insights",
            nav_chat: "🤖 AI Copilot",
            hero_text: "Upload any maize leaf capture to instantly isolate leaf diseases with spatial Grad-CAM attention heatmaps and receive calibrated fungicide dosages, causes, and prevention protocols.",
            upload_title: "Maize Leaf Diagnostic Station",
            upload_subtitle: "Select or drag & drop a clear maize leaf or crop image to isolate fungal lesions",
            drag_drop: "Drag & drop your leaf image here",
            or: "or",
            choose_image: "Choose Image",
            max_size: "Max: 5MB",
            image_ready: "Leaf Image Ready for Diagnosis",
            change_image: "Change Image",
            remove_image: "Remove",
            run_diagnosis: "Run AI Neural Diagnosis",
            analyzing: "Krishi Mitra AI is verifying leaf authenticity and computing Grad-CAM spatial heatmap...",
            mandi_badge: "APMC Mandi Maize Rates",
            moisture_rule: "Standard Moisture: <14%",
            kisan_call: "Kisan Helpline: 1800-180-1551",
            print_slip: "Print Report",
            dash_heading: "Farm Operations & AI Crop Doctor",
            dash_subheading: "Live maize canopy surveillance, micro-lesion Grad-CAM spatial heatmaps, and calibrated 16L knapsack backpack sprayer prescriptions.",
            open_bot_btn: "Open Voice Copilot 🎙️",
            metric_health: "Canopy Health Score",
            metric_scans: "Diagnoses Today",
            metric_pathogen: "Dominant Pathogen",
            metric_window: "Fungicide Spray Window",
            fert_title: "Maize Growth Stage Nutrition & Top-Dressing Schedule",
            fert_v0_tag: "Stage 1: Basal at Sowing",
            fert_v0_title: "DAP + Potash + Zinc",
            fert_v0_desc: "Apply 50 kg DAP + 25 kg MOP + 10 kg Zinc Sulfate per acre into seed furrow for deep root establishment.",
            fert_v4_tag: "Stage 2: Knee-High (30-35 Days)",
            fert_v4_title: "1st Urea Top-Dressing",
            fert_v4_desc: "Top-dress 35 kg Urea per acre when plants reach knee height. Weed before fertilizer application for optimal uptake.",
            fert_v8_tag: "Stage 3: Tasseling / Silking (50-55 Days)",
            fert_v8_title: "2nd Urea Top-Dressing",
            fert_v8_desc: "Apply remaining 30 kg Urea per acre just prior to tassel emergence. Critical irrigation required during cob fill.",
            logs_title: "Recent Crop Inspection Logs",
            rapid_title: "Rapid Field Advisory & Weather Alerts",
            tech_title: "Krishi Mitra AI Capabilities",
            chat_tooltip: "Ask Krishi Mitra Copilot 🎙️ / ಕೃಷಿ ಮಿತ್ರ",
            chat_title: "Krishi Mitra Voice Copilot",
            online: "Online • Voice & Text Assistant",
            chat_welcome: "Greetings Farmer! I am Krishi Mitra AI.",
            chat_help: "Ask me about maize diseases, 16-liter pump dosages, APMC mandi rates, fertilizer schedule, or tap the microphone 🎙️ to speak!",
            question_rust: "What causes brown rust spots?",
            question_blight: "What is Northern Leaf Blight?",
            question_fertilizer: "Recommended fertilizer ratio?",
            question_yield: "How to maximize maize yield?",
            chat_placeholder: "Ask question or tap mic to speak...",
            footer_healthy: "Healthy Maize",
            footer_yield: "Better Yields",
            footer_smart: "Smart Farming",
            footer_future: "Sustainable Future",
            footer_farmers: "for Farmers"
        },
        kn: {
            tagline: "ಸ್ಮಾರ್ಟ್ ಬೆಳೆ ವೈದ್ಯ ಮತ್ತು ನಿಖರ ಕೃಷಿ ಸಹಾಯಕ",
            nav_home: "🏠 ಮುಖಪುಟ",
            nav_scan: "🔍 ಬೆಳೆ ವೈದ್ಯ",
            nav_disease: "📖 ರೋಗ ಮಾಹಿತಿ",
            nav_care: "🌱 ಆರೈಕೆ ಸಲಹೆಗಳು",
            nav_yield: "📊 ಇಳುವರಿ ಮಾಹಿತಿ",
            nav_chat: "🤖 ಧ್ವನಿ ಸಹಾಯಕ",
            hero_text: "ಸ್ಪಷ್ಟವಾದ ಮೆಕ್ಕೆಜೋಳದ ಎಲೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ, ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯ ಹೀಟ್‌ಮ್ಯಾಪ್ ಮೂಲಕ ರೋಗ ಪತ್ತೆಹಚ್ಚಿ ಮತ್ತು 16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ ಬೇಕಾದ ನಿಖರ ಔಷಧ ಪ್ರಮಾಣ ಪಡೆಯಿರಿ.",
            upload_title: "ಮೆಕ್ಕೆಜೋಳ ಎಲೆ ರೋಗ ತಪಾಸಣಾ ಕೇಂದ್ರ",
            upload_subtitle: "ಮೆಕ್ಕೆಜೋಳದ ಎಲೆ ಅಥವಾ ಬೆಳೆಯ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆ ಮಾಡಿ",
            drag_drop: "ನಿಮ್ಮ ಎಲೆ ಚಿತ್ರವನ್ನು ಇಲ್ಲಿ ಎಳೆದು ಬಿಡಿ",
            or: "ಅಥವಾ",
            choose_image: "ಚಿತ್ರ ಆಯ್ಕೆ ಮಾಡಿ",
            max_size: "ಗರಿಷ್ಠ ಗಾತ್ರ: 5MB",
            image_ready: "ಚಿತ್ರ ಸಿದ್ಧವಾಗಿದೆ",
            change_image: "ಚಿತ್ರ ಬದಲಿಸಿ",
            remove_image: "ತೆಗೆದುಹಾಕಿ",
            run_diagnosis: "AI ಬೆಳೆ ರೋಗ ತಪಾಸಣೆ ನಡೆಸಿ",
            analyzing: "ಕೃಷಿ ಮಿತ್ರ AI ಎಲೆಯ ಆರೋಗ್ಯ ಮತ್ತು ಹೀಟ್‌ಮ್ಯಾಪ್ ಪರಿಶೀಲಿಸುತ್ತಿದೆ...",
            mandi_badge: "APMC ಮಂಡಿ ಮೆಕ್ಕೆಜೋಳ ದರ",
            moisture_rule: "ಕಾಳಿನ ತೇವಾಂಶ ಮಿತಿ: <14%",
            kisan_call: "ಕಿಸಾನ್ ಸಹಾಯವಾಣಿ: 1800-180-1551",
            print_slip: "ವರದಿ ಮುದ್ರಿಸಿ",
            dash_heading: "ಜಮೀನು ಕಾರ್ಯಾಚರಣೆ & ಬೆಳೆ ವೈದ್ಯ ಕೇಂದ್ರ",
            dash_subheading: "ನೈಜ ಸಮಯದ ಮೆಕ್ಕೆಜೋಳ ಮೇಲ್ವಿಚಾರಣೆ, Grad-CAM ಹೀಟ್‌ಮ್ಯಾಪ್ ಮತ್ತು 16 ಲೀಟರ್ ಬೆನ್ನುಹೊರೆ ಪಂಪ್ ಔಷಧ ಪ್ರಮಾಣ.",
            open_bot_btn: "ಧ್ವನಿ ಸಹಾಯಕ ತೆರೆಯಿರಿ 🎙️",
            metric_health: "ಬೆಳೆ ಆರೋಗ್ಯ ಪ್ರಮಾಣ",
            metric_scans: "ಇಂದಿನ ತಪಾಸಣೆಗಳು",
            metric_pathogen: "ಮುಖ್ಯ ರೋಗ",
            metric_window: "ಔಷಧ ಸಿಂಪಡಣೆ ಸಮಯ",
            fert_title: "ಮೆಕ್ಕೆಜೋಳ ಹಂತವಾರು ರಸಗೊಬ್ಬರ & ಮೇಲುಗೊಬ್ಬರ ವೇಳಾಪಟ್ಟಿ",
            fert_v0_tag: "ಹಂತ 1: ಬಿತ್ತನೆ ಸಮಯದಲ್ಲಿ (ಆರಂಭಿಕ)",
            fert_v0_title: "DAP + ಪೊಟ್ಯಾಶ್ + ಜಿಂಕ್",
            fert_v0_desc: "ಎಕರೆಗೆ 50 ಕೆಜಿ DAP + 25 ಕೆಜಿ MOP + 10 ಕೆಜಿ ಜಿಂಕ್ ಸಲ್ಫೇಟ್ ಸಾಲಿನಲ್ಲಿ ಹಾಕಿ ಬೀಜ ಬಿತ್ತನೆ ಮಾಡಿ.",
            fert_v4_tag: "ಹಂತ 2: ಮೊಣಕಾಲುದ್ದ (30-35 ದಿನಗಳು)",
            fert_v4_title: "1ನೇ ಬಾರಿ ಯೂರಿಯಾ ಮೇಲುಗೊಬ್ಬರ",
            fert_v4_desc: "ಎಕರೆಗೆ 35 ಕೆಜಿ ಯೂರಿಯಾವನ್ನು ಕಳೆ ಕಿತ್ತ ನಂತರ ಸಾಲುಗಳ ಪಕ್ಕದಲ್ಲಿ ಮೇಲುಗೊಬ್ಬರವಾಗಿ ನೀಡಿ.",
            fert_v8_tag: "ಹಂತ 3: ತೆನೆ/ಹೂವಾಡುವ ಹಂತ (50-55 ದಿನಗಳು)",
            fert_v8_title: "2ನೇ ಬಾರಿ ಯೂರಿಯಾ ಮೇಲುಗೊಬ್ಬರ",
            fert_v8_desc: "ತೆನೆ ಮೂಡುವ ಮುನ್ನ ಉಳಿದ 30 ಕೆಜಿ ಯೂರಿಯಾ ನೀಡಿ, ಕಾಳು ತುಂಬಲು ಸಮರ್ಪಕ ನೀರಾವರಿ ಒದಗಿಸಿ.",
            logs_title: "ಇತ್ತೀಚಿನ ಬೆಳೆ ತಪಾಸಣಾ ವಿವರಗಳು",
            rapid_title: "ತ್ವರಿತ ಕೃಷಿ ಸಲಹೆಗಳು & ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ",
            tech_title: "ಕೃಷಿ ಮಿತ್ರ AI ತಂತ್ರಜ್ಞಾನ",
            chat_tooltip: "ಕೃಷಿ ಮಿತ್ರ ಧ್ವನಿ ಸಹಾಯಕ 🎙️",
            chat_title: "ಕೃಷಿ ಮಿತ್ರ ಧ್ವನಿ ಸಹಾಯಕ",
            online: "ಆನ್‌ಲೈನ್ • ಧ್ವನಿ ಮತ್ತು ಬರಹ",
            chat_welcome: "ನಮಸ್ಕಾರ ರೈತ ಬಾಂಧವರೇ! ನಾನು ನಿಮ್ಮ ಕೃಷಿ ಮಿತ್ರ AI.",
            chat_help: "ಮೆಕ್ಕೆಜೋಳದ ರೋಗಗಳು, 16 ಲೀಟರ್ ಪಂಪ್ ಔಷಧ ಪ್ರಮಾಣ, ಮಂಡಿ ದರ, ಗೊಬ್ಬರ ಅಥವಾ ಆರೈಕೆಯ ಬಗ್ಗೆ ಕೇಳಿ. ಮಾತನಾಡಲು ಮೈಕ್ರೊಫೋನ್ 🎙️ ಒತ್ತಿ!",
            question_rust: "ಮೆಕ್ಕೆಜೋಳದ ಎಲೆಗಳಲ್ಲಿ ಕಂದು ಕಲೆಗಳು ಏಕೆ ಬರುತ್ತವೆ?",
            question_blight: "Northern Leaf Blight ಎಂದರೇನು?",
            question_fertilizer: "ಮೆಕ್ಕೆಜೋಳಕ್ಕೆ ಯಾವ ಗೊಬ್ಬರ ಸೂಕ್ತ?",
            question_yield: "ಮೆಕ್ಕೆಜೋಳದ ಇಳುವರಿಯನ್ನು ಹೇಗೆ ಹೆಚ್ಚಿಸಬಹುದು?",
            chat_placeholder: "ಪ್ರಶ್ನೆ ಕೇಳಿ ಅಥವಾ ಮಾತನಾಡಲು ಮೈಕ್ ಒತ್ತಿ...",
            footer_healthy: "ಆರೋಗ್ಯಕರ ಮೆಕ್ಕೆಜೋಳ",
            footer_yield: "ಉತ್ತಮ ಇಳುವರಿ",
            footer_smart: "ಸ್ಮಾರ್ಟ್ ಕೃಷಿ",
            footer_future: "ಸುಸ್ಥಿರ ಭವಿಷ್ಯ",
            footer_farmers: "ರೈತರಿಗಾಗಿ"
        },
        hi: {
            tagline: "स्मार्ट फसल डॉक्टर एवं सटीक कृषि सहायक",
            nav_home: "🏠 होम",
            nav_scan: "🔍 फसल डॉक्टर",
            nav_disease: "📖 रोग जानकारी",
            nav_care: "🌱 देखभाल सुझाव",
            nav_yield: "📊 उपज जानकारी",
            nav_chat: "🤖 वॉइस असिस्टेंट",
            hero_text: "मक्के की पत्ती की तस्वीर अपलोड करें, AI द्वारा रोग की पहचान करें और 16 लीटर स्प्रे पंप के अनुसार सटीक दवा की मात्रा जानें।",
            upload_title: "मक्का पत्ती रोग निदान केंद्र",
            upload_subtitle: "मक्के की पत्ती या फसल की तस्वीर चुनें",
            drag_drop: "अपनी तस्वीर यहां खींचकर छोड़ें",
            or: "या",
            choose_image: "तस्वीर चुनें",
            max_size: "अधिकतम आकार: 5MB",
            image_ready: "तस्वीर तैयार है",
            change_image: "तस्वीर बदलें",
            remove_image: "हटाएं",
            run_diagnosis: "AI फसल रोग निदान शुरू करें",
            analyzing: "कृषि मित्र AI पत्ती की जांच और हीटमैप तैयार कर रहा है...",
            mandi_badge: "APMC मंडी मक्का भाव",
            moisture_rule: "मानक नमी: <14%",
            kisan_call: "किसान हेल्पलाइन: 1800-180-1551",
            print_slip: "रिपोर्ट प्रिंट करें",
            dash_heading: "फार्म ऑपरेशंस एवं AI फसल डॉक्टर",
            dash_subheading: "मक्के की फसल की निगरानी, Grad-CAM हीटमैप और 16 लीटर नैपसैक स्प्रे पंप की सटीक दवा मात्रा।",
            open_bot_btn: "वॉइस कोपायलट खोलें 🎙️",
            metric_health: "फसल स्वास्थ्य स्कोर",
            metric_scans: "आज के कुल निदान",
            metric_pathogen: "प्रमुख रोग",
            metric_window: "छिड़काव का अनुकूल समय",
            fert_title: "मक्का फसल विकास अनुसार संतुलित खाद समय-सारणी",
            fert_v0_tag: "चरण 1: बुवाई के समय (आधार खाद)",
            fert_v0_title: "DAP + पोटाश + जिंक",
            fert_v0_desc: "50 किग्रा DAP + 25 किग्रा MOP + 10 किग्रा जिंक सल्फेट प्रति एकड़ बुवाई के समय कतारों में दें।",
            fert_v4_tag: "चरण 2: घुटने की ऊंचाई (30-35 दिन)",
            fert_v4_title: "प्रथम यूरिया टॉप-ड्रेसिंग",
            fert_v4_desc: "निराई-गुड़ाई के बाद 35 किग्रा यूरिया प्रति एकड़ पौधों की जड़ों के पास देकर सिंचाई करें।",
            fert_v8_tag: "चरण 3: फूल/मंजरी आते समय (50-55 दिन)",
            fert_v8_title: "द्वितीय यूरिया टॉप-ड्रेसिंग",
            fert_v8_desc: "नर मंजरी निकलने से ठीक पहले 30 किग्रा यूरिया प्रति एकड़ डालें। दाना भरते समय सिंचाई अनिवार्य है।",
            logs_title: "हालिया फसल निरीक्षण लॉग्स",
            rapid_title: "त्वरित कृषि सलाह व मौसम अलर्ट",
            tech_title: "कृषि मित्र AI क्षमताएं",
            chat_tooltip: "कृषि मित्र वॉइस कोपायलट 🎙️",
            chat_title: "कृषि मित्र वॉइस कोपायलट",
            online: "ऑनलाइन • आवाज और चैट",
            chat_welcome: "नमस्ते किसान भाई! मैं आपका कृषि मित्र AI हूँ।",
            chat_help: "मक्के के रोगों, 16 लीटर पंप की दवा, मंडी भाव, खाद आदि के बारे में पूछें या बोलने के लिए माइक 🎙️ दबाएं!",
            question_rust: "मक्के की पत्तियों पर भूरे धब्बे क्यों आते हैं?",
            question_blight: "Northern Leaf Blight क्या है?",
            question_fertilizer: "मक्के के लिए कौन सा खाद उपयुक्त है?",
            question_yield: "मक्के की उपज कैसे बढ़ा सकते हैं?",
            chat_placeholder: "प्रश्न लिखें या बोलने के लिए माइक दबाएं...",
            footer_healthy: "स्वस्थ मक्का",
            footer_yield: "बेहतर उपज",
            footer_smart: "स्मार्ट खेती",
            footer_future: "टिकाऊ भविष्य",
            footer_farmers: "किसानों के लिए"
        }
    };

    let currentLanguage = localStorage.getItem("agriAdvisorLanguage") || "en";
    let lastPredictionData = null;
    let selectedAcreage = 1.0;

    // =========================================================
    // 2. LANGUAGE SYSTEM CONTROLLER
    // =========================================================
    function setLanguage(lang) {
        currentLanguage = translations[lang] ? lang : "en";
        localStorage.setItem("agriAdvisorLanguage", currentLanguage);
        document.documentElement.lang = currentLanguage;

        document.querySelectorAll("[data-i18n]").forEach(function (el) {
            const key = el.getAttribute("data-i18n");
            if (translations[currentLanguage][key]) {
                el.textContent = translations[currentLanguage][key];
            }
        });

        document.querySelectorAll("[data-i18n-placeholder]").forEach(function (el) {
            const key = el.getAttribute("data-i18n-placeholder");
            if (translations[currentLanguage][key]) {
                el.placeholder = translations[currentLanguage][key];
            }
        });

        const selector = document.getElementById("languageSelector");
        if (selector) selector.value = currentLanguage;

        // If there is an existing diagnosis result, re-render it in the new language!
        if (lastPredictionData) {
            renderDiagnosis(lastPredictionData);
        }
    }

    const languageSelector = document.getElementById("languageSelector");
    if (languageSelector) {
        languageSelector.addEventListener("change", function () {
            setLanguage(this.value);
        });
    }

    setLanguage(currentLanguage);

    // =========================================================
    // 3. WEB SPEECH API: VOICE ASSISTANT (STT & TTS)
    // =========================================================

    // Text to Speech (Audio Readout)
    function speakText(textToSpeak, langCode) {
        if (!("speechSynthesis" in window)) {
            console.log("Speech synthesis not supported on this device/browser.");
            return;
        }

        window.speechSynthesis.cancel(); // Stop any active speech

        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        const code = langCode || currentLanguage;

        if (code === "kn") {
            utterance.lang = "kn-IN";
            utterance.rate = 0.95;
        } else if (code === "hi") {
            utterance.lang = "hi-IN";
            utterance.rate = 0.95;
        } else {
            utterance.lang = "en-IN";
            utterance.rate = 1.0;
        }

        window.speechSynthesis.speak(utterance);
    }

    window.stopAudioSpeech = function () {
        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
        }
    };

    // Speech to Text (Microphone Voice Input)
    const micButton = document.getElementById("micButton");
    const chatInput = document.getElementById("chatInput");
    let recognition = null;
    let isListening = false;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition && micButton) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;

        recognition.onstart = function () {
            isListening = true;
            micButton.classList.add("listening");
            micButton.title = "Listening... Speak now!";
            if (chatInput) {
                chatInput.placeholder = currentLanguage === "kn" ? "ಕೇಳಿಸಿಕೊಳ್ಳುತ್ತಿದ್ದೇನೆ... ಮಾತನಾಡಿ..." :
                                       (currentLanguage === "hi" ? "सुन रहा हूँ... बोलिए..." : "Listening... Speak now...");
            }
        };

        recognition.onresult = function (event) {
            const transcript = event.results[0][0].transcript;
            if (chatInput && transcript) {
                chatInput.value = transcript;
                sendQueryToBot(transcript);
            }
        };

        recognition.onerror = function (event) {
            console.log("Speech recognition error:", event.error);
            stopListening();
        };

        recognition.onend = function () {
            stopListening();
        };

        function stopListening() {
            isListening = false;
            if (micButton) {
                micButton.classList.remove("listening");
                micButton.title = "Speak to Krishi Mitra";
            }
            if (chatInput) {
                chatInput.placeholder = translations[currentLanguage].chat_placeholder;
            }
        }

        micButton.addEventListener("click", function () {
            if (isListening) {
                recognition.stop();
            } else {
                // Set language for recognition
                if (currentLanguage === "kn") {
                    recognition.lang = "kn-IN";
                } else if (currentLanguage === "hi") {
                    recognition.lang = "hi-IN";
                } else {
                    recognition.lang = "en-IN";
                }
                try {
                    recognition.start();
                } catch (e) {
                    console.log("Could not start recognition:", e);
                }
            }
        });
    } else if (micButton) {
        micButton.style.display = "none"; // Hide mic if not supported
    }

    // =========================================================
    // 4. IMAGE UPLOAD & DRAG-AND-DROP CONTROLLER
    // =========================================================
    const uploadArea = document.getElementById("uploadArea");
    const uploadContent = document.getElementById("uploadContent");
    const selectedImageContent = document.getElementById("selectedImageContent");
    const imageInput = document.getElementById("imageInput");
    const imagePreview = document.getElementById("imagePreview");
    const selectedFileName = document.getElementById("selectedFileName");
    const detectButton = document.getElementById("detectButton");
    const detectButtonText = document.getElementById("detectButtonText");
    const loadingSection = document.getElementById("loadingSection");
    const laserLine = document.getElementById("laserLine");
    const resultArea = document.getElementById("result");
    const removeImageButton = document.getElementById("removeImageButton");
    const removeImageButton2 = document.getElementById("removeImageButton2");
    const changeImageButton = document.getElementById("changeImageButton");

    let currentSelectedFile = null;

    function handleFileSelection(file) {
        if (!file) return;

        if (file.size > 5 * 1024 * 1024) {
            alert("Image size must be less than 5MB.");
            return;
        }

        currentSelectedFile = file;

        if (imagePreview) {
            imagePreview.src = URL.createObjectURL(file);
        }
        if (selectedFileName) {
            selectedFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        }
        if (uploadContent) uploadContent.style.display = "none";
        if (selectedImageContent) selectedImageContent.style.display = "block";

        if (detectButton) detectButton.disabled = false;
        if (resultArea) {
            resultArea.innerHTML = "";
            resultArea.style.display = "none";
        }
        lastPredictionData = null;
    }

    function resetImageSelection() {
        currentSelectedFile = null;
        lastPredictionData = null;
        window.stopAudioSpeech();

        if (imageInput) imageInput.value = "";
        if (imagePreview) imagePreview.src = "";
        if (uploadContent) uploadContent.style.display = "block";
        if (selectedImageContent) selectedImageContent.style.display = "none";
        if (detectButton) detectButton.disabled = true;
        if (laserLine) laserLine.style.display = "none";
        if (resultArea) {
            resultArea.innerHTML = "";
            resultArea.style.display = "none";
        }
    }

    if (imageInput) {
        imageInput.addEventListener("change", function () {
            if (this.files && this.files[0]) {
                handleFileSelection(this.files[0]);
            }
        });
    }

    if (uploadArea) {
        uploadArea.addEventListener("dragover", function (e) {
            e.preventDefault();
            this.classList.add("drag-over");
        });
        uploadArea.addEventListener("dragleave", function () {
            this.classList.remove("drag-over");
        });
        uploadArea.addEventListener("drop", function (e) {
            e.preventDefault();
            this.classList.remove("drag-over");
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileSelection(e.dataTransfer.files[0]);
            }
        });
    }

    if (removeImageButton) removeImageButton.addEventListener("click", resetImageSelection);
    if (removeImageButton2) removeImageButton2.addEventListener("click", resetImageSelection);
    if (changeImageButton) {
        changeImageButton.addEventListener("click", function (e) {
            e.stopPropagation();
            if (imageInput) imageInput.click();
        });
    }

    // =========================================================
    // 5. RUN AI NEURAL DIAGNOSIS (PREDICT)
    // =========================================================
    if (detectButton) {
        detectButton.addEventListener("click", async function () {
            if (!currentSelectedFile) return;

            window.stopAudioSpeech();
            detectButton.disabled = true;
            if (detectButtonText) detectButtonText.textContent = "Analyzing Neural Activation...";
            if (laserLine) laserLine.style.display = "block";
            if (loadingSection) loadingSection.style.display = "block";
            if (resultArea) {
                resultArea.innerHTML = "";
                resultArea.style.display = "none";
            }

            const formData = new FormData();
            formData.append("file", currentSelectedFile);
            formData.append("language", currentLanguage);

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();

                if (!response.ok || data.success === false) {
                    renderRejection(data.message || "The uploaded image could not be processed.");
                    return;
                }

                lastPredictionData = data;
                renderDiagnosis(data);

            } catch (err) {
                renderRejection("Could not connect to the Krishi Mitra server. Please ensure Flask is running.");
            } finally {
                if (loadingSection) loadingSection.style.display = "none";
                if (laserLine) laserLine.style.display = "none";
                if (detectButton) detectButton.disabled = false;
                if (detectButtonText) detectButtonText.textContent = "Run AI Neural Diagnosis";
            }
        });
    }

    function renderRejection(message) {
        if (!resultArea) return;
        resultArea.innerHTML = `
            <div class="rejected-result">
                <h2>❌ IMAGE NOT ACCEPTED</h2>
                <p>${escapeHtml(message)}</p>
                <p style="margin-top: 6px; font-size: 13px; color: var(--text-muted);">Please upload a clear, high-resolution photograph of a maize leaf.</p>
            </div>
        `;
        resultArea.style.display = "block";
        resultArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // =========================================================
    // 6. RENDER DIAGNOSIS: GRAD-CAM + 16L PUMP CALCULATOR + AUDIO
    // =========================================================
    function renderDiagnosis(data) {
        if (!resultArea) return;

        const isHealthy = data.disease === "healthy";
        const diseaseConf = Number(data.confidence || 0);
        const maizeConf = Number(data.maize_confidence || 0);
        const originalSrc = imagePreview ? imagePreview.src : "";

        // Get localized strings
        const langData = (data.multilingual && data.multilingual[currentLanguage]) ?
                         data.multilingual[currentLanguage] : data;

        const diseaseName = langData.name || data.disease_name || data.disease;
        const cause = langData.cause || data.cause || "";
        const symptoms = langData.symptoms || data.symptoms || "";
        const prevention = langData.prevention || data.prevention || "";
        const dosage = langData.dosage || data.dosage || "";
        const timing = langData.timing || data.timing || "";
        const yieldText = langData.yield || data.yield || "";
        const calcData = data.calc || { dose_per_16L_pump_g: 40.0, total_pack_needed_g: 500, approx_cost_inr: 200 };

        // Construct spoken summary for Voice Assistant
        const spokenText = isHealthy ?
            (currentLanguage === "kn" ? "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಯಾವುದೇ ರೋಗವಿಲ್ಲ." :
             currentLanguage === "hi" ? "आपकी फसल स्वस्थ है। कोई रोग नहीं है।" :
             "The maize crop is healthy. No disease detected.") :
            (currentLanguage === "kn" ? `ಪತ್ತೆಯಾದ ರೋಗ: ${diseaseName}. 16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ ${calcData.dose_per_16L_pump_g} ಗ್ರಾಂ ಔಷಧ ಬೆರೆಸಿ ಬೆಳಿಗ್ಗೆ ಸಿಂಪಡಿಸಿ.` :
             currentLanguage === "hi" ? `रोग पाया गया: ${diseaseName}। 16 लीटर स्प्रे पंप में ${calcData.dose_per_16L_pump_g} ग्राम दवा मिलाकर सुबह छिड़काव करें।` :
             `Disease detected: ${diseaseName}. Use ${calcData.dose_per_16L_pump_g} grams per 16 liter pump and spray in the early morning.`);

        // Heatmap Component
        const heatmapHtml = data.heatmap_image ? `
            <div class="heatmap-section">
                <div class="heatmap-section-title">
                    🔬 <span>AI Spatial Attention (Grad-CAM Neural Activation Map)</span>
                </div>
                <div class="heatmap-grid">
                    <div class="heatmap-card">
                        <div class="heatmap-card-header">📷 Uploaded Leaf Image</div>
                        <img src="${originalSrc}" alt="Uploaded Leaf">
                    </div>
                    <div class="heatmap-card">
                        <div class="heatmap-card-header">🌡️ AI Attention Heatmap (Grad-CAM Layer: conv2d_3)</div>
                        <img src="${data.heatmap_image}" alt="Grad-CAM Activation">
                    </div>
                </div>
                <p class="heatmap-note">
                    🔴 <strong>Heatmap interpretation:</strong> Warm colored regions (Red / Orange / Yellow) represent the exact areas of the leaf that activated the AI neural network to detect this condition.
                </p>
            </div>
        ` : '';

        // Pump & Acreage Calculator Component
        const totalWaterL = Math.round(selectedAcreage * 200);
        const totalPumps = (totalWaterL / 16).toFixed(1);
        const totalChemicalG = Math.round(selectedAcreage * (calcData.total_pack_needed_g || 500));
        const estimatedCost = Math.round(selectedAcreage * (calcData.approx_cost_inr || 200));

        const calcHtml = isHealthy ? '' : `
            <div class="calculator-card">
                <div class="calc-title">
                    🎒 <span>16-Liter Knapsack Sprayer & Land Dosage Calculator</span>
                </div>

                <div class="calc-acre-selector">
                    <span style="font-size: 13px; font-weight: 700; color: var(--text-secondary); margin-right: 6px;">Select Land Area:</span>
                    <button type="button" class="calc-acre-pill ${selectedAcreage === 0.5 ? 'active' : ''}" onclick="window.updateAcreage(0.5)">0.5 Acre</button>
                    <button type="button" class="calc-acre-pill ${selectedAcreage === 1.0 ? 'active' : ''}" onclick="window.updateAcreage(1.0)">1.0 Acre</button>
                    <button type="button" class="calc-acre-pill ${selectedAcreage === 2.0 ? 'active' : ''}" onclick="window.updateAcreage(2.0)">2.0 Acres</button>
                    <button type="button" class="calc-acre-pill ${selectedAcreage === 3.0 ? 'active' : ''}" onclick="window.updateAcreage(3.0)">3.0 Acres</button>
                    <button type="button" class="calc-acre-pill ${selectedAcreage === 5.0 ? 'active' : ''}" onclick="window.updateAcreage(5.0)">5.0 Acres</button>
                </div>

                <div class="calc-results-grid">
                    <div class="calc-box">
                        <div class="calc-box-label">Dose Per 16L Pump</div>
                        <div class="calc-box-val">${calcData.dose_per_16L_pump_g} g/mL</div>
                        <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">~2.5 spoons</div>
                    </div>
                    <div class="calc-box">
                        <div class="calc-box-label">Total Water Needed</div>
                        <div class="calc-box-val">${totalWaterL} Liters</div>
                        <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">Clean water</div>
                    </div>
                    <div class="calc-box">
                        <div class="calc-box-label">16L Backpack Pumps</div>
                        <div class="calc-box-val">${totalPumps} Pumps</div>
                        <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">Knapsack loads</div>
                    </div>
                    <div class="calc-box">
                        <div class="calc-box-label">Total Chemical to Buy</div>
                        <div class="calc-box-val">${totalChemicalG} grams</div>
                        <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">From retail shop</div>
                    </div>
                    <div class="calc-box">
                        <div class="calc-box-label">Estimated Cost</div>
                        <div class="calc-box-val">~₹${estimatedCost}</div>
                        <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">Approximate</div>
                    </div>
                </div>
            </div>
        `;

        // Weather Spray Advisory
        const weatherHtml = `
            <div class="weather-advisory-box">
                <span style="font-size: 24px;">🌤️</span>
                <div>
                    <strong style="color: #67e8f9; font-size: 13.5px;">Real-World Spraying Window Advisory:</strong>
                    <p style="margin: 2px 0 0 0; font-size: 12.5px; color: var(--text-secondary);">
                        Best spraying window is early morning (<strong>06:30 AM – 09:30 AM</strong>) or late afternoon. Avoid spraying if wind speed exceeds 10 km/h or if rain is forecasted within 3 hours.
                    </p>
                </div>
            </div>
        `;

        // WhatsApp Share Content
        const waText = encodeURIComponent(
            `*🌾 Krishi Mitra AI Crop Diagnosis Report*\n` +
            `*Disease:* ${diseaseName} (${data.disease})\n` +
            `*AI Confidence:* ${diseaseConf.toFixed(1)}%\n` +
            `*Recommended Spray:* ${calcData.chemical_name || 'Mancozeb 75% WP'}\n` +
            `*16L Pump Dosage:* ${calcData.dose_per_16L_pump_g}g per 16L pump (500g/acre in 200L water)\n` +
            `*Timing:* Early morning (06:30-09:30 AM)\n` +
            `Diagnosed via Krishi Mitra AI Smart Agriculture.`
        );

        // Pre-Harvest Interval (PHI) & PPE Safety Card
        const phiHtml = isHealthy ? '' : `
            <div class="phi-safety-card">
                <div class="phi-header">
                    <div class="phi-title">
                        🛡️ <span>Pre-Harvest Interval (PHI) & Chemical Safety Protocol</span>
                    </div>
                    <span class="phi-badge">⚠️ 14-Day Mandatory Cutoff</span>
                </div>
                <p class="phi-text">
                    <strong>Critical Farmer Safety Rule:</strong> Fungicide spraying must cease at least <strong>14 days prior to harvest</strong> of maize cobs to guarantee zero toxic chemical residues in food, feed, or silage.
                </p>
                <div class="ppe-checklist-grid">
                    <div class="ppe-item">
                        <span class="ppe-icon">😷</span>
                        <span>Wear N95/Face Mask</span>
                    </div>
                    <div class="ppe-item">
                        <span class="ppe-icon">🧤</span>
                        <span>Use Rubber/Nitrile Gloves</span>
                    </div>
                    <div class="ppe-item">
                        <span class="ppe-icon">💨</span>
                        <span>Spray With Wind Direction</span>
                    </div>
                    <div class="ppe-item">
                        <span class="ppe-icon">🧼</span>
                        <span>Wash Hands & Nozzle Thoroughly</span>
                    </div>
                </div>
            </div>
        `;

        resultArea.innerHTML = `
            <div class="success-result">
                <div class="result-header-badge ${isHealthy ? 'badge-healthy' : 'badge-disease'}">
                    ${isHealthy ? '✅ Normal / Healthy Canopy' : '⚠️ Pathogen Detected'}
                </div>

                <h2 class="result-title">${escapeHtml(diseaseName)}</h2>
                ${langData.pathogen ? `<div class="result-pathogen">🔬 Pathogen: <strong>${escapeHtml(langData.pathogen)}</strong></div>` : ''}

                <!-- Audio Advisory Speaker Bar -->
                <div class="audio-advisory-bar">
                    <div class="audio-advisory-info">
                        <span style="font-size: 20px;">🔊</span>
                        <div>
                            <strong style="color: #ffffff; font-size: 13.5px;">Voice Audio Advisory:</strong>
                            <p style="margin: 0; font-size: 12px; color: var(--text-emerald);">Listen in ${currentLanguage === 'kn' ? 'Kannada (ಕನ್ನಡ)' : (currentLanguage === 'hi' ? 'Hindi (हिन्दी)' : 'English')}</p>
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button type="button" class="audio-btn" onclick="window.playAudioDiagnosis()">
                            ▶️ Play Audio
                        </button>
                        <button type="button" style="background: rgba(239,68,68,0.2); border: 1px solid rgba(239,68,68,0.4); color: #f87171; border-radius: 99px; padding: 6px 12px; font-size: 12px; cursor: pointer;" onclick="window.stopAudioSpeech()">
                            ⏹️ Stop
                        </button>
                    </div>
                </div>

                <div class="metrics-row">
                    <div class="metric-box">
                        <div class="metric-label">Disease Confidence</div>
                        <div class="metric-value">${diseaseConf.toFixed(1)}%</div>
                        <div style="background: rgba(255,255,255,0.08); border-radius: 99px; height: 6px; margin-top: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #10b981, #34d399); width: ${diseaseConf.toFixed(1)}%; height: 100%; border-radius: 99px;"></div>
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Maize Verification Gate</div>
                        <div class="metric-value">${maizeConf.toFixed(1)}%</div>
                        <div style="background: rgba(255,255,255,0.08); border-radius: 99px; height: 6px; margin-top: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #06b6d4, #3b82f6); width: ${maizeConf.toFixed(1)}%; height: 100%; border-radius: 99px;"></div>
                        </div>
                    </div>
                </div>

                ${heatmapHtml}

                ${weatherHtml}

                ${calcHtml}

                ${phiHtml}

                <div class="advisory-grid">
                    ${cause ? `
                        <div class="advisory-card cause">
                            <div class="advisory-card-title">🔬 Cause & Environmental Triggers</div>
                            <p class="advisory-card-text">${escapeHtml(cause)}</p>
                        </div>
                    ` : ''}

                    ${symptoms ? `
                        <div class="advisory-card" style="border-left-color: #6366f1; background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25);">
                            <div class="advisory-card-title">👁️ Visual Symptoms on Leaf</div>
                            <p class="advisory-card-text">${escapeHtml(symptoms)}</p>
                        </div>
                    ` : ''}

                    ${prevention ? `
                        <div class="advisory-card prevention">
                            <div class="advisory-card-title">🛡️ Prevention & Agronomic Practices</div>
                            <p class="advisory-card-text">${escapeHtml(prevention)}</p>
                        </div>
                    ` : ''}

                    ${dosage ? `
                        <div class="advisory-card dosage">
                            <div class="advisory-card-title">💊 Recommended Treatment & Exact Dosage</div>
                            <div class="dosage-highlight-box">
                                <p class="advisory-card-text" style="font-weight: 600; color: #6ee7b7;">${escapeHtml(dosage)}</p>
                            </div>
                            ${timing ? `
                                <p class="advisory-card-text" style="margin-top: 10px;">
                                    <strong>⏱️ Application Timing:</strong> ${escapeHtml(timing)}
                                </p>
                            ` : ''}
                        </div>
                    ` : ''}

                    ${yieldText ? `
                        <div class="advisory-card yield">
                            <div class="advisory-card-title">📊 Yield Impact & Economic Guidance</div>
                            <p class="advisory-card-text">${escapeHtml(yieldText)}</p>
                        </div>
                    ` : ''}
                </div>

                <!-- Action Buttons: WhatsApp Share, Print, Chatbot -->
                <div class="action-buttons-row">
                    <a href="https://api.whatsapp.com/send?text=${waText}" target="_blank" rel="noopener" class="btn-whatsapp">
                        📲 Share on WhatsApp
                    </a>
                    <button type="button" onclick="window.print()" class="btn-action-outline">
                        🖨️ Print Diagnostic Slip
                    </button>
                    <button type="button" onclick="window.askCopilotAbout('${escapeHtml(diseaseName)}')" class="btn-action-outline" style="border-color: rgba(6, 182, 212, 0.4); color: #67e8f9;">
                        💬 Ask Copilot About Dosage
                    </button>
                </div>
            </div>
        `;

        resultArea.style.display = "block";
        resultArea.scrollIntoView({ behavior: "smooth", block: "nearest" });

        // Bind audio playback trigger
        window.playAudioDiagnosis = function () {
            speakText(spokenText, currentLanguage);
        };
    }

    // Helper to update acreage in pump calculator
    window.updateAcreage = function (acres) {
        selectedAcreage = acres;
        if (lastPredictionData) {
            renderDiagnosis(lastPredictionData);
        }
    };

    // =========================================================
    // 7. AI AGRI COPILOT CHATBOT
    // =========================================================
    const chatMessages = document.getElementById("chatMessages");
    const sendChatButton = document.getElementById("sendChatButton");
    const quickQuestionButtons = document.querySelectorAll(".quick-question");

    function appendChatMessage(sender, text) {
        if (!chatMessages) return null;
        const msgDiv = document.createElement("div");
        msgDiv.className = sender === "user" ? "user-message" : "bot-message";

        if (sender === "user") {
            msgDiv.innerHTML = `<div><p>${escapeHtml(text)}</p></div><span>👨‍🌾</span>`;
        } else {
            msgDiv.innerHTML = `
                <span>🌱</span>
                <div>
                    ${formatBotMessage(text)}
                    <button type="button" onclick="window.speakBotReply(this)" style="background: none; border: none; font-size: 14px; cursor: pointer; margin-top: 6px; color: var(--brand-primary-light);" title="Read out loud">
                        🔊 Read Aloud
                    </button>
                </div>
            `;
        }

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgDiv;
    }

    window.speakBotReply = function (btn) {
        const parent = btn.closest("div");
        if (parent) {
            const clone = parent.cloneNode(true);
            const btnInClone = clone.querySelector("button");
            if (btnInClone) btnInClone.remove();
            const textToSpeak = clone.textContent.trim();
            speakText(textToSpeak, currentLanguage);
        }
    };

    function formatBotMessage(text) {
        let html = escapeHtml(text);
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        const lines = html.split("\n");
        return lines.map(line => {
            const trimmed = line.trim();
            if (trimmed.startsWith("•") || trimmed.startsWith("-")) {
                return `<p style="margin: 3px 0 3px 12px;">${trimmed}</p>`;
            }
            return trimmed ? `<p style="margin: 4px 0;">${trimmed}</p>` : `<br>`;
        }).join("");
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    async function sendQueryToBot(questionText) {
        if (!questionText) return;

        appendChatMessage("user", questionText);
        const typingEl = appendChatMessage("bot", "⏳ Consulting Krishi Mitra Knowledge Base...");

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: questionText, language: currentLanguage })
            });
            const data = await response.json();

            if (response.ok && data.success && data.reply) {
                if (typingEl) {
                    typingEl.querySelector("div").innerHTML = `
                        ${formatBotMessage(data.reply)}
                        <button type="button" onclick="window.speakBotReply(this)" style="background: none; border: none; font-size: 14px; cursor: pointer; margin-top: 6px; color: var(--brand-primary-light);" title="Read out loud">
                            🔊 Read Aloud
                        </button>
                    `;
                }
                return;
            }
            throw new Error(data.reply || "Invalid server response");
        } catch (err) {
            // Client-side fallback if server is offline
            const fallbackReply = getLocalChatFallback(questionText, currentLanguage);
            if (typingEl) {
                typingEl.querySelector("div").innerHTML = `
                    ${formatBotMessage(fallbackReply)}
                    <button type="button" onclick="window.speakBotReply(this)" style="background: none; border: none; font-size: 14px; cursor: pointer; margin-top: 6px; color: var(--brand-primary-light);" title="Read out loud">
                        🔊 Read Aloud
                    </button>
                `;
            }
        }
    }

    function getLocalChatFallback(q, lang) {
        const key = q.toLowerCase();
        if (key.includes("dosage") || key.includes("dose") || key.includes("pump") || key.includes("ಔಷಧ") || key.includes("ದವಾ") || key.includes("दवा")) {
            if (lang === "kn") return "🌽 **16 ಲೀಟರ್ ಪಂಪ್ ಔಷಧ ಪ್ರಮಾಣ:** ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP @ 40 ಗ್ರಾಂ ಅಥವಾ ಪ್ರಾಪಿಕೊನಜೋಲ್ 25% EC (ಟಿಲ್ಟ್) @ 16 ಮಿ.ಲೀ ಪ್ರತಿ 16L ಪಂಪ್‌ಗೆ. ಎಕರೆಗೆ ಸುಮಾರು 12.5 ಪಂಪ್‌ಗಳು ಬೇಕು.";
            if (lang === "hi") return "🌽 **16 लीटर पंप के लिए कवकनाशी मात्रा:** मैंकोजेब 75% WP @ 40 ग्राम या प्रोपिकोनाज़ोल (टिल्ट) @ 16 मिली प्रति 16 लीटर पंप। प्रति एकड़ लगभग 12-13 पंप आवश्यक हैं।";
            return "🌽 **16L Knapsack Pump Dosage:** Mancozeb 75% WP @ 40g per 16L pump OR Propiconazole 25% EC @ 16mL per 16L pump. One acre requires ~12.5 pumps (200L water).";
        }
        if (key.includes("prevent") || key.includes("ತಡೆಗಟ್ಟು") || key.includes("रोकथाम")) {
            if (lang === "kn") return "🛡️ **ತಡೆಗಟ್ಟುವಿಕೆ:** ರೋಗ ನಿರೋಧಕ ಹೈಬ್ರಿಡ್ ಬೀಜ ಬಳಸಿ, ದ್ವಿದಳ ಧಾನ್ಯಗಳೊಂದಿಗೆ ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ, ಮತ್ತು ಸಾಲು ಅಂತರ 60x20 ಸೆಂ.ಮೀ ಕಾಪಾಡಿ.";
            if (lang === "hi") return "🛡️ **रोकथाम:** रोगरोधी संकर बीज बोएं, फसल चक्र अपनाएं और 60x20 सेमी की दूरी रखें।";
            return "🛡️ **Prevention:** Plant certified resistant hybrids, rotate fields with legumes, maintain 60x20 cm plant spacing, and deep plow post-harvest.";
        }
        if (key.includes("mandi") || key.includes("price") || key.includes("rate") || key.includes("ಮಾರುಕಟ್ಟೆ") || key.includes("ಮಂಡಿ") || key.includes("मंडी") || key.includes("भाव")) {
            if (lang === "kn") return "💰 **APMC ಮಾರುಕಟ್ಟೆ ದರ:** ಕೇಂದ್ರ MSP: ₹2,225/ಕ್ವಿಂಟಾಲ್. ದಾವಣಗೆರೆ/ರಾಣೆಬೆನ್ನೂರು ಮಂಡಿ ದರ: ₹2,280 - ₹2,410/ಕ್ವಿಂಟಾಲ್. ಕಾಳಿನಲ್ಲಿ ತೇವಾಂಶ 14% ಕ್ಕಿಂತ ಕಡಿಮೆ ಇರಲಿ.";
            if (lang === "hi") return "💰 **APMC मंडी भाव:** सरकारी MSP: ₹2,225/क्विंटल। स्थानीय मंडी मॉडल भाव: ₹2,280 से ₹2,410/क्विंटल। नमी 14% से कम रखें।";
            return "💰 **Maize Mandi Benchmark:** Govt MSP is ₹2,225/Qtl. Key APMC modal prices range between ₹2,280 – ₹2,410/Qtl. Ensure grain moisture is below 14%.";
        }
        if (key.includes("call") || key.includes("helpline") || key.includes("toll") || key.includes("ಸಹಾಯವಾಣಿ") || key.includes("हेल्पलाइन")) {
            if (lang === "kn") return "📞 **ಕಿಸಾನ್ ಸಹಾಯವಾಣಿ:** ಟೋಲ್-ಫ್ರೀ ಸಂಖ್ಯೆ: 1800-180-1551 (ಬೆಳಗ್ಗೆ 6:00 ರಿಂದ ರಾತ್ರಿ 10:00 ರವರೆಗೆ ಉಚಿತ). ಕನ್ನಡದಲ್ಲೇ ನೇರ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ.";
            if (lang === "hi") return "📞 **किसान हेल्पलाइन:** टोल-फ्री नंबर: 1800-180-1551 (सुबह 6:00 से रात 10:00 बजे तक निःशुल्क)। सीधे कृषि वैज्ञानिकों से बात करें।";
            return "📞 **Kisan National Helpline:** Toll-free 1800-180-1551 (6:00 AM – 10:00 PM). Connects directly to certified agricultural scientists in your language.";
        }
        if (key.includes("fertiliz") || key.includes("urea") || key.includes("dap") || key.includes("ಗೊಬ್ಬರ") || key.includes("खाद")) {
            if (lang === "kn") return "🌾 **ರಸಗೊಬ್ಬರ ವೇಳಾಪಟ್ಟಿ:** ಬಿತ್ತನೆಗೆ DAP 50 ಕೆಜಿ + ಜಿಂಕ್ 10 ಕೆಜಿ. 30 ದಿನಕ್ಕೆ ಯೂರಿಯಾ 35 ಕೆಜಿ ಮೇಲುಗೊಬ್ಬರ. ತೆನೆ ಬರುವಾಗ 30 ಕೆಜಿ ಯೂರಿಯಾ ನೀಡಿ.";
            if (lang === "hi") return "🌾 **खाद समय-सारणी:** बुवाई पर 50 किग्रा DAP + 10 किग्रा जिंक। 30-35 दिन पर 35 किग्रा यूरिया टॉप-ड्रेसिंग। फूल आने पर 30 किग्रा यूरिया दें।";
            return "🌾 **Fertilizer Protocol:** Basal at sowing: DAP 50kg + Zinc 10kg/acre. Knee-high (30 days): 35kg Urea top-dressing. Tasseling: 30kg Urea + irrigation.";
        }
        return "I can assist you with maize disease causes, prevention, 16L knapsack sprayer dosages, APMC mandi rates, Kisan helpline, and fertilizers. Please ask a question or tap the mic!";
    }

    if (quickQuestionButtons) {
        quickQuestionButtons.forEach(btn => {
            btn.addEventListener("click", function () {
                const questionType = this.getAttribute("data-question");
                let qText = this.textContent.trim();
                if (questionType === "dosage") {
                    qText = currentLanguage === "kn" ? "16 ಲೀಟರ್ ಪಂಪ್ ಔಷಧ ಪ್ರಮಾಣ ತಿಳಿಸಿ" : (currentLanguage === "hi" ? "16 लीटर पंप की दवा की मात्रा बताएं" : "16L knapsack pump dosage?");
                } else if (questionType === "prevent") {
                    qText = currentLanguage === "kn" ? "ಮೆಕ್ಕೆಜೋಳ ರೋಗ ತಡೆಗಟ್ಟುವ ಕ್ರಮಗಳು" : (currentLanguage === "hi" ? "मक्के के रोगों की रोकथाम" : "How to prevent maize diseases?");
                } else if (questionType === "mandi") {
                    qText = currentLanguage === "kn" ? "ಮೆಕ್ಕೆಜೋಳ ಮಂಡಿ ಬೆಲೆ ಎಷ್ಟು?" : (currentLanguage === "hi" ? "मक्के का मंडी भाव क्या है?" : "What is current maize APMC mandi price?");
                } else if (questionType === "helpline") {
                    qText = currentLanguage === "kn" ? "ಕಿಸಾನ್ ಕಾಲ್ ಸೆಂಟರ್ ಸಹಾಯವಾಣಿ ಸಂಖ್ಯೆ" : (currentLanguage === "hi" ? "किसान हेल्पलाइन नंबर" : "Kisan Call Center helpline number?");
                } else if (questionType === "fertilizer") {
                    qText = currentLanguage === "kn" ? "ಮೆಕ್ಕೆಜೋಳಕ್ಕೆ ಯೂರಿಯಾ ಗೊಬ್ಬರ ಯಾವಾಗ ಹಾಕಬೇಕು?" : (currentLanguage === "hi" ? "मक्के में यूरिया खाद कब डालें?" : "When to apply urea top-dressing for maize?");
                } else if (questionType === "safety") {
                    qText = currentLanguage === "kn" ? "ಔಷಧ ಸಿಂಪಡಣೆ ಸುರಕ್ಷತೆ ಮತ್ತು PHI ನಿಯಮ" : (currentLanguage === "hi" ? "स्प्रे सुरक्षा और PHI नियम" : "Pesticide spray safety and PHI guidelines?");
                }
                sendQueryToBot(qText);
            });
        });
    }

    function handleChatSubmit() {
        if (!chatInput) return;
        const text = chatInput.value.trim();
        if (!text) return;
        chatInput.value = "";
        sendQueryToBot(text);
    }

    if (sendChatButton) sendChatButton.addEventListener("click", handleChatSubmit);
    if (chatInput) {
        chatInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") handleChatSubmit();
        });
    }

    // =========================================================
    // FLOATING CHAT DRAWER CONTROLLER
    // =========================================================
    const floatingChatDrawer = document.getElementById("floatingChatDrawer");

    window.toggleFloatingChat = function () {
        if (!floatingChatDrawer) return;
        floatingChatDrawer.classList.toggle("open");
        if (floatingChatDrawer.classList.contains("open")) {
            if (chatInput) chatInput.focus();
            if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    };

    window.openFloatingChat = function () {
        if (!floatingChatDrawer) return;
        floatingChatDrawer.classList.add("open");
        if (chatInput) chatInput.focus();
        if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    window.closeFloatingChat = function () {
        if (!floatingChatDrawer) return;
        floatingChatDrawer.classList.remove("open");
    };

    window.askCopilotAbout = function (diseaseName) {
        window.openFloatingChat();
        if (!chatInput) return;
        const query = currentLanguage === 'kn' ? `${diseaseName} ರೋಗಕ್ಕೆ 16 ಲೀಟರ್ ಪಂಪ್ ಔಷಧ ಪ್ರಮಾಣ ಮತ್ತು ಪರಿಹಾರವೇನು?` :
                     (currentLanguage === 'hi' ? `${diseaseName} रोग के लिए 16 लीटर पंप की दवा की मात्रा क्या है?` :
                      `What is the 16L pump dosage and prevention for ${diseaseName}?`);
        chatInput.value = query;
        sendQueryToBot(query);
    };

    console.log("[Krishi Mitra AI] Voice Copilot & Pump Calculator loaded successfully.");
});
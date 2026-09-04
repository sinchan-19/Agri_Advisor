/* =========================================================
   LOGIN
   ========================================================= */

function openLogin() {

    document
        .getElementById("loginOverlay")
        .classList.add("active");

}


function closeLogin() {

    document
        .getElementById("loginOverlay")
        .classList.remove("active");

}


/* =========================================================
   PASSWORD VISIBILITY
   ========================================================= */

function togglePassword() {

    const password =
        document.getElementById("password");

    if (!password) {
        return;
    }

    if (password.type === "password") {

        password.type = "text";

    } else {

        password.type = "password";

    }

}


/* =========================================================
   LOGIN
   ========================================================= */

function login() {

    const usernameElement =
        document.getElementById("username");

    const passwordElement =
        document.getElementById("password");

    if (!usernameElement || !passwordElement) {
        return;
    }

    const username =
        usernameElement.value;

    const password =
        passwordElement.value;


    if (
        username === "admin" &&
        password === "1234"
    ) {

        alert("Login successful!");

        closeLogin();

        window.location.href =
            "dashboard.html";

    } else {

        alert(
            "Incorrect username or password."
        );

    }

}


/* =========================================================
   LOGOUT
   ========================================================= */

function logout() {

    window.location.href =
        "index.html";

}


/* =========================================================
   MAIZE DISEASE DETECTION
   ========================================================= */

async function detectDisease() {

    console.log("");
    console.log("====================================");
    console.log("MAIZE AI DETECTION STARTED");
    console.log("====================================");


    /* -----------------------------------------------------
       GET HTML ELEMENTS
       ----------------------------------------------------- */

    const fileInput =
        document.getElementById("maizeImage");

    const resultDiv =
        document.getElementById("result");


    /* -----------------------------------------------------
       CHECK RESULT ELEMENT
       ----------------------------------------------------- */

    if (!resultDiv) {

        console.error(
            "ERROR: #result element not found."
        );

        alert(
            "Result display area was not found."
        );

        return;

    }


    /* -----------------------------------------------------
       CHECK FILE INPUT
       ----------------------------------------------------- */

    if (!fileInput) {

        console.error(
            "ERROR: #maizeImage element not found."
        );

        resultDiv.innerHTML =
            `
            <div class="rejected-result">
                <h2>Image Upload Error</h2>
                <p>Image upload control was not found.</p>
            </div>
            `;

        return;

    }


    /* -----------------------------------------------------
       CHECK IMAGE
       ----------------------------------------------------- */

    if (
        !fileInput.files ||
        fileInput.files.length === 0
    ) {

        alert(
            "Please select a maize leaf image first."
        );

        return;

    }


    /* -----------------------------------------------------
       GET IMAGE
       ----------------------------------------------------- */

    const file =
        fileInput.files[0];


    console.log(
        "Selected image:",
        file.name
    );

    console.log(
        "Image type:",
        file.type
    );

    console.log(
        "Image size:",
        file.size,
        "bytes"
    );


    /* -----------------------------------------------------
       CHECK IMAGE TYPE
       ----------------------------------------------------- */

    const allowedTypes = [

        "image/jpeg",
        "image/png",
        "image/webp"

    ];


    if (!allowedTypes.includes(file.type)) {

        resultDiv.innerHTML =
            `
            <div class="rejected-result">

                <h2>❌ INVALID IMAGE</h2>

                <p>
                    Please upload a JPG, JPEG,
                    PNG or WEBP image.
                </p>

            </div>
            `;

        return;

    }


    /* -----------------------------------------------------
       PREPARE FORM DATA
       ----------------------------------------------------- */

    const formData =
        new FormData();


    formData.append(
        "file",
        file
    );


    /* -----------------------------------------------------
       SHOW LOADING
       ----------------------------------------------------- */

    resultDiv.innerHTML =
        `
        <div class="loading-result">

            <h2>🔄 AI is analyzing the image...</h2>

            <p>
                First checking whether the image
                is a maize image.
            </p>

            <p>
                Please wait...
            </p>

        </div>
        `;


    /* -----------------------------------------------------
       SEND TO FLASK
       ----------------------------------------------------- */

    try {

        console.log("");
        console.log(
            "Sending image to /predict..."
        );


        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        console.log(
            "HTTP status:",
            response.status
        );


        /* -------------------------------------------------
           READ RESPONSE
           ------------------------------------------------- */

        let data;


        try {

            data =
                await response.json();

        } catch (jsonError) {

            console.error(
                "Could not read JSON response:",
                jsonError
            );


            resultDiv.innerHTML =
                `
                <div class="rejected-result">

                    <h2>❌ SERVER ERROR</h2>

                    <p>
                        The server returned an
                        invalid response.
                    </p>

                </div>
                `;

            return;

        }


        console.log("");
        console.log(
            "===================================="
        );

        console.log(
            "FLASK RESPONSE:"
        );

        console.log(
            data
        );

        console.log(
            "===================================="
        );


        /* =================================================
           MOST IMPORTANT SAFETY CHECK
           
           If Flask says is_maize === false,
           STOP HERE.

           DO NOT DISPLAY DISEASE.
           ================================================= */

        if (
            data.is_maize === false
        ) {

            console.log("");
            console.log(
                "❌ NON-MAIZE IMAGE"
            );

            console.log(
                "Disease result will NOT be displayed."
            );


            const message =
                data.error ||
                "This image is not recognized as a maize image.";


            resultDiv.innerHTML =
                `
                <div class="rejected-result">

                    <h2>
                        ❌ NON-MAIZE IMAGE
                    </h2>

                    <p>
                        ${message}
                    </p>

                    <p>
                        <strong>
                            Disease detection was not performed.
                        </strong>
                    </p>

                    <p>
                        Please upload a clear maize
                        leaf or maize crop image.
                    </p>

                </div>
                `;


            /* ---------------------------------------------
               CRITICAL:
               STOP FUNCTION COMPLETELY
               --------------------------------------------- */

            return;

        }


        /* =================================================
           SECOND SAFETY CHECK

           If server explicitly says rejected,
           don't display disease.
           ================================================= */

        if (
            data.rejected === true
        ) {

            console.log("");
            console.log(
                "❌ IMAGE REJECTED"
            );


            const message =
                data.error ||
                "The image could not be accepted.";


            resultDiv.innerHTML =
                `
                <div class="rejected-result">

                    <h2>
                        ❌ IMAGE REJECTED
                    </h2>

                    <p>
                        ${message}
                    </p>

                    <p>
                        Please upload a clear maize
                        leaf image.
                    </p>

                </div>
                `;


            return;

        }


        /* =================================================
           CHECK SERVER ERROR
           ================================================= */

        if (!response.ok) {

            console.error(
                "Server returned error:",
                data
            );


            resultDiv.innerHTML =
                `
                <div class="rejected-result">

                    <h2>
                        ❌ SERVER ERROR
                    </h2>

                    <p>
                        ${data.error ||
                        "Something went wrong while processing the image."}
                    </p>

                </div>
                `;


            return;

        }


        /* =================================================
           ONLY NOW DISPLAY DISEASE RESULT
           
           This means:
           
           is_maize === true
           AND
           rejected !== true
           ================================================= */

        if (
            data.is_maize === true &&
            data.success === true
        ) {

            console.log("");
            console.log(
                "✅ MAIZE IMAGE ACCEPTED"
            );

            console.log(
                "Disease:",
                data.disease
            );

            console.log(
                "Disease confidence:",
                data.confidence,
                "%"
            );


            const disease =
                data.disease ||
                data.prediction ||
                "Unknown";


            const confidence =
                Number(
                    data.confidence || 0
                );


            const gateConfidence =
                Number(
                    data.gate_confidence || 0
                );


            /* ---------------------------------------------
               DISEASE DISPLAY
               --------------------------------------------- */

            resultDiv.innerHTML =
                `
                <div class="success-result">

                    <h2>
                        🌽 MAIZE IMAGE DETECTED
                    </h2>

                    <p>
                        <strong>Disease:</strong>
                        ${disease}
                    </p>

                    <p>
                        <strong>
                            Disease Confidence:
                        </strong>
                        ${confidence.toFixed(2)}%
                    </p>

                    <p>
                        <strong>
                            Maize Gate Confidence:
                        </strong>
                        ${gateConfidence.toFixed(2)}%
                    </p>

                </div>
                `;


            return;

        }


        /* =================================================
           UNKNOWN RESPONSE
           ================================================= */

        console.warn(
            "Unknown response received:",
            data
        );


        resultDiv.innerHTML =
            `
            <div class="rejected-result">

                <h2>
                    ❌ IMAGE NOT ACCEPTED
                </h2>

                <p>
                    The AI could not safely identify
                    this image.
                </p>

                <p>
                    Please upload a clear maize
                    leaf image.
                </p>

            </div>
            `;


    } catch (error) {


        /* =================================================
           NETWORK / FETCH ERROR
           ================================================= */

        console.error(
            "Prediction request failed:",
            error
        );


        resultDiv.innerHTML =
            `
            <div class="rejected-result">

                <h2>
                    ❌ CONNECTION ERROR
                </h2>

                <p>
                    Could not connect to the
                    Agri Advisor AI server.
                </p>

                <p>
                    Make sure Flask is running.
                </p>

            </div>
            `;

    }


}
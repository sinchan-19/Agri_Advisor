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


function togglePassword() {

    const password =
        document.getElementById("password");

    if (password.type === "password") {

        password.type = "text";

    } else {

        password.type = "password";

    }

}


function login() {

    const username =
        document.getElementById("username").value;

    const password =
        document.getElementById("password").value;


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
function logout() {

    window.location.href = "index.html";

}
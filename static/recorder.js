var buttonRecord = document.getElementById("record");
var buttonStop = document.getElementById("stop");
var buttonTrain = document.getElementById("training")

buttonStop.disabled = true;

buttonRecord.onclick = function () {
    var train_name = document.getElementById("name").value;
    if(train_name !== ""){
    // var url = window.location.href + "record_status";
    buttonRecord.disabled = true;
    buttonStop.disabled = false;
    buttonTrain.disabled = true;
    // disable download link
    var downloadLink = document.getElementById("download");
    downloadLink.text = "";
    downloadLink.href = "";

    
    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    }
    xhr.open("POST", "/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({
        status: "true",
        train_name: train_name
    }));
} else {alert('Vui lòng nhập tên !')} 
};

buttonStop.onclick = function () {
    buttonRecord.disabled = false;
    buttonStop.disabled = true;
    buttonTrain.disabled = false;
    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
            document.getElementById("name").value = "";
            // enable download link
            var downloadLink = document.getElementById("download");
            downloadLink.text = "Video Recorded !";
        }
    }
    xhr.open("POST", "/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({
        status: "false"
    }));
};

$(document).ready(function () {
    $('#training').click(function () {
        $("#trainFinish").hide()
        $('#training').prop('disabled', true);
        $.ajax({
            url: "/start_training",
            type: 'post',
        });

        var eventSource = new EventSource("/listen");
        eventSource.addEventListener("message", function (e) {
            console.log(e.data)
        }, false);

        eventSource.addEventListener("online", function (e) {
            $("#trainProcess").show()
            data = JSON.parse(e.data);
            $('#process').html(data.process);
            if (data.process === "Done!"){
                $("#trainProcess").hide()
                $("#trainFinish").show()
                eventSource.close()
                $('#training').prop('disabled', false);
            }
        }, true);
    })
})
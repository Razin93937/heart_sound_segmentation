import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'

var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#519872',
    progressColor: '#A6DEC0',
    responsive: true,
});

const wsRegions = wavesurfer.registerPlugin(RegionsPlugin.create())

var colors = ["#FFFFFF70", "#06d6a000", "#FFF20070", "#06d6a000"]
var labels = ['s1', 'systole', 's2', 'diastole']

$(document).ready(function() {
    $.get('/models', function(data) {
        data.forEach(function(model) {
            $('#modelSelect').append('<option value="' + model + '">' + model + '</option>');
        });
    });
});

$('#modelSelect').on('change', function() {
    if ($(this).val() && $('#audioupload')[0].files.length > 0) {
        $('#submit').prop('disabled', false);
    } else {
        $('#submit').prop('disabled', true);
    }
});

$('#playpause').on('click', function (event) {
    wavesurfer.playPause();
});

$('#audioupload').on('change', function (event) {
    if ($(this).val() && $('#modelSelect').val()) {
        $('#submit').prop('disabled', false);
    } else {
        $('#submit').prop('disabled', true);
    }
    $('.play-btn').prop('disabled', false);

    var file = event.target.files[0];
    $('#label').html(file.name);
    var reader = new FileReader();
    reader.onload = function (event) {
        var blob = new window.Blob([new Uint8Array(event.target.result)], { type: file.type });
        wavesurfer.loadBlob(blob);
    };
    reader.readAsArrayBuffer(file);
    wsRegions.clearRegions();
});

$('#submit').on('click', function () {
    $('body').addClass('busy');
    $('#submit').prop('disabled', true);

    var file = $('#audioupload')[0].files[0];
    var formData = new FormData();
    formData.append('audio', file);
    formData.append('model', $('#modelSelect').val());
    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (r) {
            var r2 = []
            for (let i = 0; i < r['start'].length; i++) {
                r2.push({ start: r['start'][i], end: r['end'][i], label: labels[r['label'][i]] })
                wsRegions.addRegion({
                    start: r['start'][i],
                    end: r['end'][i],
                    loop: false,
                    drag: false,
                    resize: false,
                    color: colors[r['label'][i]]
                });
            }
            $('#timestamps').bootstrapTable('load', r2);
            $('body').removeClass('busy');
            $(".wavesurfer-region").addClass("rounded-pill");
        }
    });
});
$(document).ready(function(){
	$('.btn-select').on('click', function(){
		$('.select').click();
	});

	$('.btn-submit').on('click', function(){
		$('.submit').click();
		$('.main-menu, h1, .upload, .text, .image').hide();
		$('.page-loading').show();
	});

	// Function for getting image size
	function getFileSize(file){
		if(file.size > (1024 * 1024)) return Math.round(file.size / (1024 * 1024)) + ' MB';
		if(file.size > 1024) return Math.round(file.size / 1024) + ' KB';

		return file.size + ' B';
	};

	$('.select').on('change', function(){
		const preview = document.querySelector('.image-upload');
		const file = document.querySelector('.select').files[0];
		const reader = new FileReader();

		reader.addEventListener('load', function(){
			// convert image file to base64 string
			preview.src = reader.result;
		}, false);

		if(file){
			reader.readAsDataURL(file);
		}

		document.querySelector('.custom-text').textContent = document.querySelector('.select').value.split('\\').pop().split('/').pop();
		document.querySelector('.custom-size').textContent = getFileSize(file);
	});
});

// Return filename + extension from a input type = file
// chooseBtn.value.split('\\').pop().split('/').pop()
// Return filename + extension from a given path
// console.log(path.split('/').pop())

var app = new Vue({
	delimiters: ['[[', ']]'],
	el: '#app',
	data: {
		changed: false,
	},
	methods: {
		checkInput: function(){
			if(this.changed == false){
				this.changed = true;
			}
		}
	}
});

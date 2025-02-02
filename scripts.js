let currentIndex = 0;
const slides = document.querySelectorAll('.slide');
const totalSlides = slides.length;

const prevButton = document.createElement('button');
prevButton.classList.add('prev-btn');
prevButton.innerHTML = '&#8592;';
document.querySelector('.image-slider').appendChild(prevButton);

const nextButton = document.createElement('button');
nextButton.classList.add('next-btn');
nextButton.innerHTML = '&#8594;';
document.querySelector('.image-slider').appendChild(nextButton);

function changeSlide() {
    currentIndex = (currentIndex + 1) % totalSlides;
    const sliderContainer = document.querySelector('.slider-container');
    sliderContainer.style.transform = `translateX(-${currentIndex * 100}%)`;
}

function prevSlide() {
    currentIndex = (currentIndex - 1 + totalSlides) % totalSlides;
    const sliderContainer = document.querySelector('.slider-container');
    sliderContainer.style.transform = `translateX(-${currentIndex * 100}%)`;
}

function nextSlide() {
    changeSlide();
}

prevButton.addEventListener('click', prevSlide);
nextButton.addEventListener('click', nextSlide);

setInterval(changeSlide, 5000);

const style = document.createElement('style');
style.innerHTML = `
    .prev-btn,
    .next-btn {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        border: none;
        font-size: 2rem;
        padding: 10px;
        cursor: pointer;
        z-index: 10;
    }

    .prev-btn {
        left: 10px;
    }

    .next-btn {
        right: 10px;
    }

    .prev-btn:hover,
    .next-btn:hover {
        background-color: rgba(0, 0, 0, 0.7);
    }
`;
document.head.appendChild(style);

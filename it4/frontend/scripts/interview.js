document.addEventListener('DOMContentLoaded', () => {
    const interviewForm = document.getElementById('interviewForm');
    const motivationInput = interviewForm.querySelector('textarea[name="motivation"]');
    const habitsInput = interviewForm.querySelector('textarea[name="habits"]');
    const followUpContainer = document.getElementById('followUpQuestions');
    const followUpContent = document.getElementById('followUpContent');
    const saveFollowUpButton = document.getElementById('saveFollowUp');
    const bookNowButton = document.getElementById('bookNow'); // Use existing "Book Now" button
    const charCountElements = document.querySelectorAll('.char-count');
    const MAX_CHARS = 250;

    function updateCharCount(textarea, countElement) {
        const currentLength = textarea.value.length;
        countElement.textContent = currentLength;
        
        if (currentLength > MAX_CHARS) {
            countElement.classList.add('text-red-600');
            textarea.classList.add('border-red-500');
        } else {
            countElement.classList.remove('text-red-600');
            textarea.classList.remove('border-red-500');
        }
    }

    const motivationCharCount = motivationInput.closest('div').nextElementSibling.querySelector('.char-count');
    motivationInput.addEventListener('input', () => {
        updateCharCount(motivationInput, motivationCharCount);
    });

    const habitsCharCount = habitsInput.closest('div').nextElementSibling.querySelector('.char-count');
    habitsInput.addEventListener('input', () => {
        updateCharCount(habitsInput, habitsCharCount);
    });

    function createFollowUpQuestionElement(question, index) {
        return `
          <div class="follow-up-question space-y-2 p-4 bg-gray-50 rounded-lg">
              <label class="block text-sm font-medium text-gray-700">
                  ${question}
              </label>
              <textarea 
                  name="followup_${index}"
                  rows="3"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md resize-none focus:ring-red-500 focus:border-red-500"
                  placeholder="Your answer..."
              ></textarea>
              <div class="flex justify-end mt-1">
                  <span class="text-sm text-gray-500">
                      <span class="char-count">0</span>/${MAX_CHARS}
                  </span>
              </div>
          </div>
      `;
    }

    function displayFollowUpQuestions(questions) {
        followUpContent.innerHTML = questions
            .map((question, index) => createFollowUpQuestionElement(question, index))
            .join('');

        followUpContent.querySelectorAll('textarea').forEach(textarea => {
            const countElement = textarea.closest('.follow-up-question').querySelector('.char-count');
            textarea.addEventListener('input', () => {
                updateCharCount(textarea, countElement);
            });
        });

        followUpContainer.classList.remove('hidden');
    }

    async function saveFollowUpAnswers() {
        const answers = Array.from(followUpContent.querySelectorAll('textarea'))
            .map(textarea => textarea.value);
    
        try {
            const response = await fetch('http://127.0.0.1:3003/api/user/interview-followup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                },
                body: JSON.stringify({ answers })
            });
    
            if (response.ok) {
                const result = await response.json();
                alert('Follow-up answers saved successfully!');
                showBookNowButton(); // Show "Book Now" button
            } else {
                const errorData = await response.json();
                alert(`Failed to save follow-up answers: ${errorData.message}`);
            }
        } catch (error) {
            console.error('Error saving follow-up answers:', error);
            alert('An error occurred while saving follow-up answers.');
        }
    }

    function showBookNowButton() {
        bookNowButton.classList.remove('hidden'); // Show the button
    }

    interviewForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        const motivationLength = motivationInput.value.length;
        const habitsLength = habitsInput.value.length;
        
        if (motivationLength > MAX_CHARS || habitsLength > MAX_CHARS) {
            alert(`Please keep your answers under ${MAX_CHARS} characters.`);
            return;
        }

        const motivation = motivationInput.value;
        const habits = habitsInput.value;

        try {
            const response = await fetch('http://127.0.0.1:3003/api/user/interview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                },
                body: JSON.stringify({
                    motivation,
                    habits
                })
            });

            if (response.ok) {
                const result = await response.json();
                alert(result.message);

                if (result.followUpQuestions) {
                    displayFollowUpQuestions(result.followUpQuestions);
                }
            } else {
                const errorData = await response.json();
                alert(`Failed to save answers: ${errorData.message}`);
            }
        } catch (error) {
            console.error('Error saving answers:', error);
            alert('An error occurred. Please try again later.');
        }
    });

    saveFollowUpButton.addEventListener('click', saveFollowUpAnswers);
});


async function downloadAnswers() {
    try {
        const token = localStorage.getItem('authToken'); // Changed from 'token' to 'authToken' to match other functions
        if (!token) {
            alert('Please login first');
            return;
        }

        const response = await fetch('http://127.0.0.1:3003/api/user/interview', { // Updated to match backend route
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to fetch interview data');
        }

        const data = await response.json();
        
        if (!data.success || !data.interview) {
            throw new Error('No interview data found');
        }

        // Format the data
        const interview = data.interview;
        let csvContent = 'Question,Answer\n';
        
        // Add main questions
        csvContent += `"What is your motivation for pursuing MBA?","${interview.motivation?.replace(/"/g, '""') || ''}"\n`;
        csvContent += `"Describe your habit of leadership","${interview.habits?.replace(/"/g, '""') || ''}"\n`;
        
        // Add follow-up questions if they exist
        if (interview.followUpQuestions && interview.followUpQuestions.length > 0) {
            interview.followUpQuestions.forEach(qa => {
                csvContent += `"${qa.question.replace(/"/g, '""')}","${qa.answer.replace(/"/g, '""')}"\n`;
            });
        }

        // Create and trigger download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'interview_answers.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (error) {
        console.error('Download error:', error);
        alert('Error downloading answers: ' + error.message);
    }
}


